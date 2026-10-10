# PolicyDiff

Local privacy-policy comparison by Shivam Bhartiya and Amit Tiparadi. This is a fresh implementation; the standalone Assignment 2 script and superseded sourcing/evaluation scripts have been removed.

## Run on Windows

From this project folder, in PowerShell:

```powershell
python -m venv .venv  # only if the environment does not already exist
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m scripts.setup_models
.venv/Scripts/python.exe -m scripts.source_data
.venv/Scripts/python.exe -m streamlit run app.py
```

Open http://localhost:8501. Upload earlier and later versions of the same English policy, or choose **Try a sample**. TXT, HTML and Markdown are supported, up to 5 MB and 600 clauses per policy. Compare policies, inspect the clause evidence, and translate changed clauses into Hindi. The sidebar also exposes the measured evaluation results and the PDF report.

Initial setup downloads pinned models and NLTK resources into `.cache/`. Runtime inference uses local files only. Policy text is not sent to a hosted inference service. Missing models cause a visible error, with no silent substitution. The Hindi model uses pinned original PyTorch weights and the restricted weight loader; PyTorch >=2.6 is required. CPU inference and initial downloads can take time.

`requirements-lock.txt` records the tested Python 3.12 environment; `requirements.txt` supplies supported version ranges for other compatible environments. Model IDs and revisions are pinned in `policydiff/settings.py`.

## Submission artifacts

- `submission/PolicyDiff_Report.pdf`: methodology, datasets, models, implementation contribution, evaluation protocol, measured comparisons, ablations, error analysis, Hindi evaluation, figures and references; identity fields are blank.
- `submission/PolicyDiff_Report.md`: editable report content.
- `submission/DEMO_GUIDE.md`: demonstration and viva guide.
- `submission/results/`: comparative CSV/JSON, per-class precision/recall/F1, confusion counts, timings, threshold grid, error cases, translation references/outputs and corpus/token experiments.
- `submission/figures/`: workflow, scientific plots and actual browser screenshots.
- `submission/PolicyDiff_Submission.zip`: compact source-and-evidence package. Virtual environment, model cache and the full PrivaSeer demo are excluded; 25 sampled policies are included.

The report has a neutral academic layout because the official SIT Word template was not present. It does not claim to reproduce that template.

## Methodology

1. Load text, remove presentation markup/frontmatter/change logs, normalize whitespace and smart quotes, retain numbers and grammatical context, and segment sentences under section headings.
2. Lock exact clauses as a multiset, including repeated and reordered clauses.
3. Encode residual intact clauses using `all-MiniLM-L6-v2` and cosine similarity. Long clauses are split into bounded token chunks, with re-tokenization checks and weighted pooling to avoid silent truncation.
4. Use global one-to-one Hungarian assignment with dummy unmatched nodes, a minimum similarity threshold and a small section-name bonus.
5. Classify unchanged, likely reworded, modified, added, removed or review. Competing matches trigger review. Number, unit, negation, modal and privacy-topic checks can override likely paraphrase classification.
6. Rank extracted change evidence using TextRank, prioritizing explicit signals; show at most five summary items, with all clauses available separately.
7. Calculate Flesch Reading Ease and Flesch-Kincaid grade on grammatical prose. Translate both sides of changed clauses with pretrained `Helsinki-NLP/opus-mt-en-hi`, retaining English beside Hindi. Export JSON, CSV and HTML.

`policydiff.text.token_experiments()` contains the maintained POS/stemming/lemmatization experiment. Tagging happens before filtering. Numbers and negation are retained, smart quotes are normalized explicitly, and Penn tags map to WordNet POS. Stems and filtered tokens are not fed into SBERT or readability calculations.

Similarity is evidence rather than proof of equivalence. Readability measures prose difficulty, not privacy protection. Split/merge structures and tables remain documented limitations.

## Data sourcing

Downloaded Google HTML has embedded Google API keys redacted before storage. Policy paragraphs are preserved. Provenance distinguishes the original response hash (`raw_sha256`) from the sanitized file hash (`stored_html_sha256`). Dataset snapshots and submission evidence are excluded from GitHub's code-language statistics through `.gitattributes`.

The original local PrivaSeer demo contains 4,000 TXT and 4,000 HTML snapshots. It is used for preprocessing/readability exploration, not as old/new version pairs. `data/corpus_sample/` includes the first 25 loadable English documents in filename order and their hashes. The convenience sample is not representative of the full corpus.

Seven real historical comparisons are in `data/historical/`: three Google archive pairs and two pairs each from Mozilla and Automattic's publisher-maintained Git histories. Google dates are effective dates. Repository dates are Git author timestamps for the last file commit before year-end 2022, 2023 and 2024, not necessarily effective dates. Provenance records raw URLs/commit SHAs, retrieval dates and SHA-256 hashes. Third-party policy text retains its source rights.

```powershell
.venv/Scripts/python.exe -m scripts.source_data
# Uses existing files after hash verification; avoids replacing annotated snapshots.
# --refresh deliberately fetches/replaces snapshots. Review annotations if source hashes change.
.venv/Scripts/python.exe -m scripts.corpus_profile
```

Princeton WebTAP and translation fine-tuning remain optional future extensions. The demo under `data/examples/` is explicitly synthetic.

## Evaluation and evidence authority

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m scripts.verify_project
.venv/Scripts/python.exe -m scripts.benchmark
.venv/Scripts/python.exe -m scripts.stress_eval
.venv/Scripts/python.exe -m scripts.translation_eval
.venv/Scripts/python.exe -m scripts.corpus_profile
```

Google pairs are development data; Mozilla and Automattic are held out by publisher. The same 15-point threshold grid, alignment and classification rules apply to difflib, TF-IDF, SBERT alone and PolicyDiff. Baselines disable critical-wording checks. Semantic defaults selected on development data are match 0.55, reword 0.94, ambiguity margin 0.04. Historical comparisons and fixed-threshold ablation are separate from the nine constructed stress cases.

The benchmark contains source-linked AI-reviewed references, rationale and exact-text anchors. **Independent human validation is pending.** These results measure reference agreement, not independently established real-world accuracy. Five explicit split/merge/table/heading groups are excluded from primary one-to-one metrics and retained in the manifest. Incorrect alignments outside those groups are penalized using `no_pair` on the union of reference/predicted correspondences.

Held-out macro F1: difflib 0.615, TF-IDF 0.596, SBERT alone 0.694, PolicyDiff 0.694. Critical checks have no measured incremental historical F1 benefit at the selected thresholds. On the separate nine authored stress cases, checks yield 8/9 correct versus 5/9 without checks at 0.94; this is mechanism evidence, not a generalization claim. Timings exclude model load, readability, translation and UI.

Hindi evaluation uses 15 aligned paragraphs from Google's published Hindi policy, rather than model-generated reference translations. Pretrained-model scores are SacreBLEU 16.23 (`intl`) and chrF 38.10; signatures, source hashes and all outputs are provided. Paragraph alignment was AI-reviewed; independent bilingual adequacy review is pending. The demo shows a poorly rendered/omitted account-closure condition, so reference overlap and preserved digits do not establish preserved meaning.

To obtain human-reviewed results, review `data/benchmark/*.csv` and fill `human_label` for all non-exact rows. Correct correspondence decisions in `reviewed_decisions.json` if necessary, and document reviewer identity, disagreements and resolution. Excluded groups need a future group-aware protocol. Re-running materialization preserves human labels for unchanged correspondence keys.

```powershell
.venv/Scripts/python.exe -m scripts.benchmark --human-reviewed
# Refuses incomplete human labels rather than silently mixing them with AI labels.
```

Tests verify software behavior, not research accuracy. The test suite currently has 27 passing tests. `verify_project` checks cached models, Hindi output, source hashes and clause coverage for all seven historical pairs with all three engines. Final evidence is copied to `submission/results/integration_check.json`.

## Rebuild report and package

Core evaluation requires `requirements-lock.txt`. Reporting additionally needs Matplotlib, Pillow and ReportLab:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-report.txt
.venv/Scripts/python.exe -m scripts.build_submission
.venv/Scripts/python.exe -m scripts.package_submission
```

Report generation reads saved measured results and verified screenshots. It does not rerun models or invent results. The screenshots are included so the report can be rebuilt after extraction. For new results, rerun evaluation first. Run the heavyweight model jobs sequentially on Windows.

## CLI and file conventions

```powershell
.venv/Scripts/python.exe -m policydiff data/examples/old.txt data/examples/new.txt --hindi
.venv/Scripts/python.exe -m policydiff data/examples/old.txt data/examples/new.txt --engine textual
```

Outputs go to `outputs/` unless `--output` specifies another JSON path; an HTML report is written alongside it. JSON/annotation IDs are zero-based; UI labels are one-based. Text offsets refer to cleaned loaded text rather than raw HTML bytes. CSV exports sanitize formula-like values. The final preview server is stopped after verification; start it with the run command when needed.

## Primary references

- [PrivaSeer corpus paper](https://aclanthology.org/2021.acl-long.532/)
- [PrivaSeer source repository](https://git.psu.edu/hlt-lab/PrivaSeer-Corpus)
- [Sentence-BERT paper](https://aclanthology.org/D19-1410/)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [TextRank paper](https://aclanthology.org/W04-3252/)
- [English-Hindi model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi)
- [Google policy archive](https://policies.google.com/privacy/archive)
- [Mozilla policy history](https://github.com/mozilla/legal-docs)
- [Automattic policy history](https://github.com/Automattic/legalmattic)
- [Google Hindi reference](https://policies.google.com/privacy/archive/20240328?hl=hi)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)

