# PolicyDiff

PolicyDiff compares two versions of an English privacy policy to identify added, removed, reworded, and meaningfully modified clauses. It combines semantic similarity, clause alignment, extractive summaries, readability analysis, and optional Hindi translation in a local web interface.

## Features

- Upload TXT, HTML, or Markdown policies, or explore included examples.
- Compare clauses using sentence embeddings, TF-IDF, or a textual baseline.
- Inspect changes to numbers, durations, negation, modality, and privacy topics.
- Read a concise change summary and compare Flesch readability scores.
- Translate changed clauses into Hindi with the English text alongside them.
- Export results as JSON, CSV, or HTML from the web interface.
- Reproduce baseline comparisons, ablations, and translation evaluation.

## Quick start

The tested environment uses Python 3.12. From the project directory, run these commands in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m scripts.setup_models
.venv/Scripts/python.exe -m scripts.source_data
.venv/Scripts/python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Stop the server with `Ctrl+C` in the terminal.

Initial setup downloads pinned models and NLTK resources into `.cache/` and requires an internet connection. Model inference subsequently runs locally. CPU inference and initial downloads can take time. The full PrivaSeer corpus is not needed to use the app.

`requirements-lock.txt` records the tested dependency versions. `requirements.txt` provides version ranges for other compatible environments. Model identifiers and revisions are pinned in `policydiff/settings.py`.

## Using the web interface

1. Upload earlier and later versions of the same English privacy policy, or select **Try a sample**.
2. Choose the semantic engine for embedding-based comparison, or TF-IDF/textual engines for lexical baselines.
3. Run the comparison and inspect the summary, readability scores, and individual clause matches.
4. Review the evidence for each classification and any ambiguous matches.
5. Enable Hindi translation for changed clauses when needed, then download the results.

Each policy supports up to **5 MB** of input and **600 clauses**. Supported formats are `.txt`, `.html`, and `.md`.

| Classification | Meaning |
| --- | --- |
| Unchanged | Clause text matches after normalization. |
| Reworded | A matched clause has high similarity and no detected critical wording change. |
| Modified | A matched clause falls below the rewording threshold or triggers a wording check. |
| Added | A clause appears only in the later policy. |
| Removed | A clause appears only in the earlier policy. |
| Review | Competing matches make alignment uncertain. |

## Command-line usage

```powershell
# Semantic comparison with Hindi translation
.venv/Scripts/python.exe -m policydiff data/examples/old.txt data/examples/new.txt --hindi

# Lexical baselines
.venv/Scripts/python.exe -m policydiff data/examples/old.txt data/examples/new.txt --engine tfidf
.venv/Scripts/python.exe -m policydiff data/examples/old.txt data/examples/new.txt --engine textual

# Custom output location
.venv/Scripts/python.exe -m policydiff data/examples/old.txt data/examples/new.txt --output outputs/my_comparison.json
```

The CLI writes JSON and an accompanying HTML report. The default output is `outputs/comparison.json`. JSON clause IDs are zero-based; web interface labels are one-based. Text offsets refer to cleaned policy text.

## How it works

1. **Preprocess:** Remove presentation markup, normalize whitespace and smart quotes, and segment clauses under section headings. Preserve numbers, negation, and grammatical context.
2. **Match exact clauses:** Account for repeated and reordered clauses before semantic alignment.
3. **Encode and align:** Use `all-MiniLM-L6-v2` embeddings and cosine similarity, followed by global one-to-one Hungarian matching. Long clauses use bounded token chunks and weighted pooling.
4. **Classify changes:** Apply similarity thresholds, ambiguity checks, and checks for numbers, units, negation, modality, and privacy topics.
5. **Summarize and explain:** Rank extracted evidence with TextRank and calculate Flesch Reading Ease and Flesch-Kincaid Grade Level.
6. **Translate:** Use pretrained `Helsinki-NLP/opus-mt-en-hi` to translate changed clauses while retaining their English source.

The implementation uses Streamlit, Sentence Transformers, scikit-learn, SciPy, NLTK, BeautifulSoup, textstat, and Hugging Face Transformers. Semantic defaults selected on development data are a match threshold of `0.55`, a rewording threshold of `0.94`, and an ambiguity margin of `0.04`.

## Datasets

- **PrivaSeer:** 25 sampled English policies for preprocessing and readability exploration in `data/corpus_sample/`.
- **Historical policy pairs:** Seven comparisons from Google, Mozilla, and Automattic in `data/historical/`, with source URLs, dates, and SHA-256 provenance.
- **Hindi references:** 15 aligned paragraphs from Google's published Hindi privacy policy for translation evaluation.
- **Synthetic examples:** Demonstration policies in `data/examples/` and nine constructed stress cases for wording checks.

Google archive dates are policy effective dates. Mozilla and Automattic snapshot dates are Git author timestamps and may differ from effective dates. Third-party policy text retains its source rights.

```powershell
# Verify cached historical sources and download missing sources
.venv/Scripts/python.exe -m scripts.source_data

# Profile the included PrivaSeer sample
.venv/Scripts/python.exe -m scripts.corpus_profile
```

`scripts.source_data --refresh` replaces cached snapshots; review annotations if source hashes change. Downloaded Google HTML has embedded Google API keys redacted before storage. Provenance distinguishes original response hashes from sanitized file hashes. Dataset and report files are excluded from GitHub's code-language statistics.

## Evaluation

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.verify_project
.venv/Scripts/python.exe -m scripts.benchmark
.venv/Scripts/python.exe -m scripts.stress_eval
.venv/Scripts/python.exe -m scripts.translation_eval
```

Run heavyweight model jobs sequentially on Windows. Google pairs are used for development; Mozilla and Automattic are held out by publisher. Methods share the same threshold grid and alignment rules, with critical wording checks disabled for the baselines.

| Method | Held-out macro F1 |
| --- | ---: |
| Textual baseline (difflib) | 0.615 |
| TF-IDF | 0.596 |
| SBERT alone | 0.694 |
| PolicyDiff | 0.694 |

These results measure agreement with **AI-reviewed references; independent human validation is pending**. Five documented split, merge, table, or heading groups are excluded from primary one-to-one metrics. Critical wording checks show no incremental historical F1 gain at the selected thresholds. On the separate nine constructed stress cases, they improve correct classifications from 5/9 to 8/9.

Hindi evaluation yields **SacreBLEU 16.23** and **chrF 38.10** on the 15-paragraph reference set. This is a single-publisher diagnostic; independent bilingual adequacy review is pending.

For human-reviewed evaluation, fill `human_label` for every non-exact row in `data/benchmark/*.csv`, review correspondence decisions in `reviewed_decisions.json`, and run:

```powershell
.venv/Scripts/python.exe -m scripts.benchmark --human-reviewed
```

The command rejects incomplete human labels. Detailed metrics, error cases, translation outputs, and integration checks are available in [`submission/results/`](submission/results/).

## Optional English–Hindi fine-tuning

Use the [Colab notebook](https://colab.research.google.com/github/AmitTiparadi/PolicyDiff/blob/main/notebooks/translation_finetuning.ipynb) to fine-tune on 3,000 IIT Bombay pairs for 1–3 epochs and compare both models on the same held-out test set. See the [workflow guide](notebooks/README.md) for setup and limitations.

A completed two-epoch T4 run used 3,000 training, 300 validation, and 500 test pairs. Validation loss selected the epoch-1 checkpoint. Scores recomputed from exported predictions were:

| Model | IIT Bombay BLEU | IIT Bombay chrF |
| --- | ---: | ---: |
| Pretrained | 10.41 | 32.09 |
| Fine-tuned | 10.04 | 31.12 |

Fine-tuning reduced BLEU by **0.37 points** on this sample. Validation loss changed from 4.3742 before training to 4.3209 after epoch 1 and 4.3297 after epoch 2. This is a measured negative result, not evidence of improvement. These general-domain scores are separate from the Google policy diagnostic above. One seed and a small convenience sample do not establish statistical significance or privacy-domain quality; original pretraining overlap is unknown.

The [experiment evidence](submission/results/iitb_3000_seed42/) includes exact sampled pairs, predictions, curves, configuration, and [verification results](submission/results/iitb_3000_seed42/verification.json). The app continues to use the pretrained model. Fine-tuned weights were not included in the evidence export.

## Reports and project structure

| Path | Contents |
| --- | --- |
| `app.py` | Streamlit web interface |
| `policydiff/` | Preprocessing, models, comparison logic, and exports |
| `scripts/` | Model setup, sourcing, evaluation, and report generation |
| `tests/` | Automated behavior and regression tests |
| `data/` | Example policies, historical snapshots, and benchmark references |
| `submission/results/` | Saved evaluation results |
| `submission/figures/` | Plots and interface screenshots |
| `outputs/` | Locally generated comparison exports |

Read the [technical report](submission/PolicyDiff_Report.pdf) or its [editable Markdown version](submission/PolicyDiff_Report.md) for methodology and detailed results.

To rebuild the report and local source package from saved results:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-report.txt
.venv/Scripts/python.exe -m scripts.build_submission
.venv/Scripts/python.exe -m scripts.package_submission
```

Rerun evaluation first when results need updating. Packaging excludes the virtual environment, model cache, generated comparison outputs, and full PrivaSeer corpus, and checks inputs for common credential patterns.

## Limitations

Semantic similarity does not guarantee equivalent meaning. One-to-one matching can struggle with split or merged clauses and tables. Wording checks can produce false positives, while changes to named entities can be missed. Readability scores measure prose difficulty rather than privacy protection.

Hindi translation can omit or distort conditions. Keep the English text available and review translations before relying on their meaning. The included evaluation sets are small and require independent review before making broader accuracy claims.

## References

- [PrivaSeer corpus](https://git.psu.edu/hlt-lab/PrivaSeer-Corpus)
- [Sentence-BERT](https://aclanthology.org/D19-1410/)
- [MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [TextRank](https://aclanthology.org/W04-3252/)
- [English-Hindi translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi)
- [Google policy archive](https://policies.google.com/privacy/archive)
- [Mozilla policy history](https://github.com/mozilla/legal-docs)
- [Automattic policy history](https://github.com/Automattic/legalmattic)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
