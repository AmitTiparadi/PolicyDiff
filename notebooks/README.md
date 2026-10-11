# English–Hindi fine-tuning

Open [the Colab notebook](https://colab.research.google.com/github/AmitTiparadi/PolicyDiff/blob/translation-finetuning/notebooks/translation_finetuning.ipynb), select **Runtime → Change runtime type → T4 GPU**, and run the cells in order. GPU availability and runtime limits depend on Colab; completion on a free runtime is not guaranteed.

Defaults: 3,000 training pairs, 300 validation pairs, 500 test pairs, two epochs, batch size 4 with four-step gradient accumulation, maximum sequence length 128, and learning rate 2e-5. Set `EPOCHS` to 1–3 before starting. If memory is insufficient, start a fresh run with batch size 2 and accumulation 8. The script refuses to overwrite a nonempty output directory.

The workflow uses the official splits of [CFILT's IIT Bombay English–Hindi corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi), pinned to an immutable revision. Empty, duplicate, overlapping, and overlength examples are excluded. All official validation/test English and Hindi text is reserved before selecting training pairs. Training uses a bounded shuffle buffer, so this is a convenience sample rather than a uniform sample of the full corpus. The manifest and JSONL files record the exact selected examples.

It fine-tunes the pinned `Helsinki-NLP/opus-mt-en-hi` model; this is **fine-tuning, not training from random initialization**. The pretrained model's original training overlap with IIT Bombay is unknown. “Held out” means excluded from this fine-tuning run, not necessarily unseen during pretraining.

The pretrained baseline and the checkpoint with the lowest validation loss are evaluated on identical test sentences with identical beam search and SacreBLEU settings. Test scores never select the checkpoint. Report zero or negative BLEU changes as measured; do not rerun seeds or choose epochs based on test BLEU. Inspect the validation curve and translations to discuss possible explanations rather than claiming a cause from scores alone.

## Outputs

- `comparison.json` / `comparison.csv`: paired BLEU and chrF scores, decoding-cap counts, timing, and validation losses.
- `loss_history.json` / `loss_curve.png`: logged training and validation losses.
- `bleu_comparison.png`: pretrained versus fine-tuned test BLEU.
- `test_predictions.csv`: both models' translations alongside the reference.
- `run_config.json`, `data_manifest.json`, and split JSONL files: versions, settings, hashes, and actual samples.
- `best_model/`: the selected model and tokenizer, loadable with Transformers `from_pretrained`.
- `RESULTS.md`: a report populated only after an actual completed run.

Download the evidence ZIP before disconnecting. Colab's `/content` is temporary. The evidence ZIP excludes large model/checkpoint directories; use the optional model-download cell if you want to retain the fine-tuned weights. Intermediate checkpoints can occupy several GB. Disconnections are not automatically resumed by this workflow.

The [completed seed-42 run](../experiments/iitb_3000_seed42/) trained for two epochs on a T4. Its selected epoch-1 checkpoint scored BLEU 10.04 versus pretrained 10.41, and chrF 31.12 versus 32.09. Exported predictions reproduce those scores. The app continues to use the pretrained model; the supplied evidence excludes model weights. IIT Bombay BLEU is a general-domain evaluation and does not replace the privacy-policy diagnostic.

Verify the exported evidence without loading model weights:

```powershell
.venv/Scripts/python.exe -m scripts.verify_training_evidence experiments/iitb_3000_seed42
```

For a local GPU run:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-training.txt
.venv/Scripts/python.exe -m scripts.finetune_translation --epochs 2
```

The default output folder is `outputs/translation_finetuning/`, which is ignored by Git. Copy only reviewed evidence into the repository; keep model weights out of Git.
