"""Reproducible IIT Bombay fine-tuning and paired held-out MT evaluation.

Run in a Colab GPU runtime; no results are supplied until this experiment runs.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
from time import perf_counter
import unicodedata

from policydiff.settings import TRANSLATION_MODEL, TRANSLATION_REVISION

DATASET = 'cfilt/iitb-english-hindi'
DATASET_REVISION = '321516f50bdcc1214fa75164c545478976ed84bd'


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()


def fingerprint(rows):
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def select_rows(records, tokenizer, count, max_length, blocked_en=None, blocked_hi=None):
    """Reject duplicates/overlap and overlength pairs instead of truncating references."""
    seen_en, seen_hi = set(blocked_en or ()), set(blocked_hi or ())
    selected = []
    rejected = {'empty': 0, 'overlap_or_duplicate': 0, 'overlength': 0}
    for record in records:
        pair = record['translation']
        en, hi = pair['en'].strip(), pair['hi'].strip()
        a, b = normalize(en), normalize(hi)
        if not a or not b:
            rejected['empty'] += 1
            continue
        if a in seen_en or b in seen_hi:
            rejected['overlap_or_duplicate'] += 1
            continue
        encoded = tokenizer(en, text_target=hi, truncation=False)
        if max(len(encoded['input_ids']), len(encoded['labels'])) > max_length:
            rejected['overlength'] += 1
            continue
        seen_en.add(a)
        seen_hi.add(b)
        selected.append({'en': en, 'hi': hi})
        if len(selected) == count:
            return selected, rejected
    raise ValueError(f'Only {len(selected)} eligible pairs found; requested {count}. Reduce the sample size.')


def assert_disjoint(splits):
    for field in ['en', 'hi']:
        seen = set()
        for name, rows in splits.items():
            values = [normalize(r[field]) for r in rows]
            if len(values) != len(set(values)) or seen.intersection(values):
                raise ValueError(f'Duplicate or overlapping {field} text in {name}')
            seen.update(values)


def prepare_splits(tokenizer, args):
    from datasets import load_dataset
    def stream(split):
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/data/{split}-00000-of-00001.parquet'
        return load_dataset('parquet', data_files={split: url}, split=split, streaming=True)
    # Reserve all official evaluation text before sampling training examples.
    official_test, official_validation = list(stream('test')), list(stream('validation'))
    reserved_en = {normalize(r['translation']['en']) for r in official_test + official_validation}
    reserved_hi = {normalize(r['translation']['hi']) for r in official_test + official_validation}
    random.Random(args.seed).shuffle(official_test)
    random.Random(args.seed + 1).shuffle(official_validation)
    test, test_rejections = select_rows(official_test, tokenizer, args.test_size, args.max_length)
    validation, val_rejections = select_rows(
        official_validation, tokenizer, args.validation_size, args.max_length,
        {normalize(r['translation']['en']) for r in official_test},
        {normalize(r['translation']['hi']) for r in official_test})
    # A bounded-buffer shuffle keeps RAM use small. This is not a uniform full-corpus sample.
    training = stream('train').shuffle(seed=args.seed, buffer_size=10000)
    train, train_rejections = select_rows(training, tokenizer, args.train_size, args.max_length,
                                         reserved_en, reserved_hi)
    splits = {'train': train, 'validation': validation, 'test': test}
    assert_disjoint(splits)
    manifest = {
        'dataset': DATASET, 'revision': DATASET_REVISION, 'seed': args.seed,
        'selection': 'Official splits; train streaming shuffle buffer 10000; evaluation splits shuffled in memory; first eligible rows.',
        'filters': 'Empty, duplicate English OR Hindi, cross-split overlap, or either token sequence exceeding max_length rejected.',
        'training_exclusions': 'All official validation and test English/Hindi texts, not just selected samples.',
        'max_length': args.max_length,
        'splits': {name: {'count': len(rows), 'sha256': fingerprint(rows)} for name, rows in splits.items()},
        'rejections': {'train': train_rejections, 'validation': val_rejections, 'test': test_rejections},
    }
    return splits, manifest


def score_translations(predictions, references):
    from sacrebleu.metrics import BLEU, CHRF
    bleu, chrf = BLEU(tokenize='intl'), CHRF()
    return {'bleu': bleu.corpus_score(predictions, [references]).score,
            'chrf': chrf.corpus_score(predictions, [references]).score,
            'bleu_signature': str(bleu.get_signature()), 'chrf_signature': str(chrf.get_signature())}


def generate(model, tokenizer, sources, batch_size, max_length, beams):
    import torch
    model.eval()
    predictions, capped = [], 0
    for start in range(0, len(sources), batch_size):
        batch = tokenizer(sources[start:start + batch_size], return_tensors='pt', padding=True, truncation=False)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model.generate(**batch, num_beams=beams, do_sample=False,
                                    max_new_tokens=max_length, use_cache=True)
        for row in output.tolist():
            # Skip the decoder start token; EOS can also be the start token in some models.
            capped += int(tokenizer.eos_token_id not in row[1:])
        predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
        print(f'Translated {len(predictions)}/{len(sources)}', flush=True)
    return predictions, capped


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_plots(folder, history, comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [('loss', 'Training loss'), ('eval_loss', 'Validation loss')]:
        rows = [r for r in history if key in r and 'epoch' in r]
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], marker='o', label=label)
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='IIT Bombay fine-tuning')
    ax.legend(); fig.tight_layout(); fig.savefig(folder / 'loss_curve.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Pretrained', 'Fine-tuned']
    scores = [comparison['pretrained']['bleu'], comparison['fine_tuned']['bleu']]
    bars = ax.bar(labels, scores, color=['#64748b', '#0f766e'])
    ax.bar_label(bars, fmt='%.2f')
    ax.set(ylabel='SacreBLEU (intl)', title='Identical held-out test set', ylim=(0, max(scores) * 1.2 + 1))
    fig.tight_layout(); fig.savefig(folder / 'bleu_comparison.png', dpi=160); plt.close(fig)


def run(args):
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('Select a Colab GPU runtime, or explicitly pass --allow-cpu for a slow local run.')
    folder = args.output.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Output directory is not empty. Choose a new output path to preserve previous evidence.')
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL, revision=TRANSLATION_REVISION)
    splits, manifest = prepare_splits(tokenizer, args)
    write_json(folder / 'data_manifest.json', manifest)
    for name, rows in splits.items():
        with (folder / f'{name}.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(folder / 'run_config.json', {
        **vars(args), 'output': str(folder), 'model': TRANSLATION_MODEL,
        'model_revision': TRANSLATION_REVISION, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'datasets', 'accelerate', 'sacrebleu']},
    })
    def encode(batch):
        return tokenizer(batch['en'], text_target=batch['hi'], truncation=False)
    train = Dataset.from_list(splits['train']).map(encode, batched=True, remove_columns=['en', 'hi'])
    validation = Dataset.from_list(splits['validation']).map(encode, batched=True, remove_columns=['en', 'hi'])
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL, revision=TRANSLATION_REVISION,
                                               use_safetensors=False, weights_only=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    sources = [r['en'] for r in splits['test']]
    references = [r['hi'] for r in splits['test']]
    start = perf_counter()
    baseline, baseline_capped = generate(model, tokenizer, sources, args.batch_size, args.max_length, args.beams)
    baseline_scores = score_translations(baseline, references)
    baseline_scores.update({'seconds': perf_counter() - start, 'outputs_without_eos': baseline_capped})
    write_json(folder / 'pretrained_results.json', baseline_scores)
    write_json(folder / 'pretrained_predictions.json', baseline)
    model.config.use_cache = False
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(folder / 'checkpoints'), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation_steps,
        eval_strategy='epoch', save_strategy='epoch', logging_strategy='steps', logging_steps=10,
        logging_first_step=True, load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, save_total_limit=2, report_to='none', push_to_hub=False,
        seed=args.seed, data_seed=args.seed, fp16=False, bf16=False,
        gradient_checkpointing=True, optim='adamw_torch', warmup_ratio=0.1,
        weight_decay=0.01, predict_with_generate=False, dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=validation,
                            processing_class=tokenizer,
                            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model))
    initial_validation = trainer.evaluate()
    start = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - start
    model.config.use_cache = True
    trainer.save_model(str(folder / 'best_model'))
    tokenizer.save_pretrained(folder / 'best_model')
    write_json(folder / 'loss_history.json', trainer.state.log_history)
    final_validation = trainer.evaluate()
    start = perf_counter()
    predictions, capped = generate(model, tokenizer, sources, args.batch_size, args.max_length, args.beams)
    tuned_scores = score_translations(predictions, references)
    tuned_scores.update({'seconds': perf_counter() - start, 'outputs_without_eos': capped})
    comparison = {
        'status': 'completed', 'test_count': len(sources), 'test_sha256': manifest['splits']['test']['sha256'],
        'pretrained': baseline_scores, 'fine_tuned': tuned_scores,
        'bleu_delta': tuned_scores['bleu'] - baseline_scores['bleu'],
        'initial_validation_loss': initial_validation['eval_loss'],
        'best_validation_loss': final_validation['eval_loss'],
        'selected_checkpoint': trainer.state.best_model_checkpoint, 'training_seconds': training_seconds,
        'selection_rule': 'Minimum validation loss; test scores are not used for checkpoint selection.',
        'limitations': [
            'Small, length-filtered sample with bounded-buffer training shuffle; not representative of the whole corpus.',
            'General-domain IIT Bombay results do not establish privacy-policy translation quality.',
            'Unknown overlap with the original pretrained model training data; held out from this fine-tuning run only.',
            'One seed; BLEU/chrF differences do not establish statistical significance or preserved legal meaning.',
        ],
    }
    write_json(folder / 'comparison.json', comparison)
    with (folder / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['model', 'bleu', 'chrf', 'outputs_without_eos'])
        for name in ['pretrained', 'fine_tuned']:
            row = comparison[name]; writer.writerow([name, row['bleu'], row['chrf'], row['outputs_without_eos']])
    with (folder / 'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['english', 'reference_hindi', 'pretrained_hindi', 'fine_tuned_hindi'])
        for row in zip(sources, references, baseline, predictions):
            writer.writerow(["'" + v if v.startswith(('=', '+', '-', '@')) else v for v in row])
    save_plots(folder, trainer.state.log_history, comparison)
    delta = comparison['bleu_delta']
    writeup = (f'# Translation fine-tuning results\n\n'
               f'Trained on {len(splits["train"])} IIT Bombay pairs for {args.epochs} epochs. '
               f'Checkpoint selected using {len(splits["validation"])} validation pairs.\n\n'
               f'On the same {len(sources)} held-out pairs, pretrained BLEU was {baseline_scores["bleu"]:.2f} '
               f'and fine-tuned BLEU was {tuned_scores["bleu"]:.2f} (difference {delta:+.2f}).\n\n'
               'A positive difference indicates higher reference overlap on this sample; a zero or negative '
               'difference means fine-tuning did not improve BLEU on this sample. Possible explanations '
               'to investigate include sample size, domain mismatch, and overfitting; the scores alone '
               'do not identify the cause. Compare validation loss and individual translations.\n\n'
               + '\n'.join('- ' + note for note in comparison['limitations']) + '\n')
    (folder / 'RESULTS.md').write_text(writeup, encoding='utf-8')
    print(json.dumps(comparison, indent=2), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/translation_finetuning'))
    p.add_argument('--train-size', type=int, default=3000)
    p.add_argument('--validation-size', type=int, default=300)
    p.add_argument('--test-size', type=int, default=500)
    p.add_argument('--epochs', type=int, choices=[1, 2, 3], default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accumulation-steps', type=int, default=4)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--beams', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--allow-cpu', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    for key in ['train_size', 'validation_size', 'test_size', 'batch_size', 'accumulation_steps', 'max_length', 'beams', 'learning_rate']:
        if getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive')
    run(args)
