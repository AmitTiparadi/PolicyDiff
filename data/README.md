# Policy sources

- `../PrivaSeer-Corpus-main-demo/demo/`: existing corpus, untouched; preprocessing and policy diversity reference.
- `examples/`: synthetic before/after policies, deliberately designed to demonstrate numbers, negation, rewording, additions, and removals.
- `historical/google-20190122-20191015/`: Google January 22, 2019 → October 15, 2019.
- `historical/google-20230701-20231004/`: Google July 1, 2023 → October 4, 2023.
- `historical/google-20240328-20240916/`: Google March 28, 2024 → September 16, 2024.

Each historical folder includes original HTML, extracted Markdown, and `provenance.json`. The downloader verifies the effective date printed by the publisher. Provenance includes source URLs, effective dates, retrieval time, and SHA-256 hashes for raw and extracted files.

The Google pairs were sourced through `python -m scripts.source_archives` from https://policies.google.com/privacy/archive because the Princeton corpus commit-history API returned HTTP 500. These files are publisher archive samples, not Princeton or PrivaSeer data. Three pairs from one publisher are a starter integration dataset, not a representative evaluation benchmark. They remain unannotated.

Original policy text belongs to its publisher. Samples are retained locally for academic comparison, with source attribution. Do not treat generated predictions or the synthetic demo as manually annotated ground truth.
