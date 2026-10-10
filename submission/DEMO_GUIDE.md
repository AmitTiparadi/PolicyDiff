# PolicyDiff demonstration and viva guide

Start the app with `.venv/Scripts/python.exe -m streamlit run app.py`, then open http://localhost:8501.

1. Upload `data/examples/old.txt` and `new.txt` or select the synthetic sample. Explain that it is constructed, not historical evidence.
2. Compare semantically. Show the retention change from 30 to 90 days, the removed negation, an added encryption clause, a removed contact clause and a paraphrase.
3. Inspect clause evidence: before/after highlights, similarity, signals, sections and text offsets. Similarity is not a confidence probability.
4. Translate into Hindi. Keep English visible; demonstrate that negation and digits are useful to inspect. Mention the observed poor translation of the account-closure condition rather than claiming verified legal equivalence.
5. Show readability and export JSON/CSV/HTML. A simpler policy is not necessarily more privacy-protective.
6. Select the Mozilla 2022-2023 historical pair. Explain the publisher source, snapshot dates and provenance hashes. Show the changed retention period and processor list.
7. Open Project evaluation in the sidebar and the PDF report. Explain development/test publisher separation, the baseline table, confusion matrix and ablations.

## What the contribution is

An auditable integration of semantic representations, exact anchors, global clause alignment, critical-wording checks, uncertainty review and bilingual evidence in a local upload interface. Existing pretrained models and established algorithms are reused; no newly trained model or invented algorithm is claimed.

## Questions to prepare for

- **Why retain numbers?** Duration, age and payment limits can change while sentence embeddings remain close.
- **Why avoid stopword stripping for SBERT?** Negation, modality and grammatical context carry meaning. The token experiment is a separate lab demonstration.
- **Why global alignment?** It avoids assigning the same clause to multiple earlier clauses; unmatched nodes permit additions/removals.
- **How were thresholds selected?** A common grid on three Google development pairs. Two other publishers are held out. The development set is small.
- **Does PolicyDiff outperform everything?** Historical macro F1 is 0.694 versus difflib 0.615 and TF-IDF 0.596. SBERT alone also reaches 0.694. Checks add no historical F1 gain here, though constructed stress cases show their mechanism.
- **Are the labels human ground truth?** No. Exact anchors plus AI-reviewed source correspondences are explicitly marked. Blank human-review fields and a strict re-evaluation mode are provided.
- **What are the failures?** Named processor changes, equivalent split/merge structures, Markdown tables, conservative modal checks and Hindi adequacy. The report includes examples.
- **Why BLEU and chrF?** They provide reproducible word/character overlap against published Hindi references. They do not prove adequacy or legal meaning preservation.
- **Was Hindi fine-tuned?** No. The pinned pretrained model is the first baseline, as requested.
- **Why not compare PrivaSeer files as versions?** Different snapshots/sites are not evidence of a longitudinal pair. PrivaSeer demonstrates preprocessing; historical publisher files demonstrate comparison.

## Submission checks

Fill the blank identity fields in the report. Its neutral layout does not reproduce the unavailable official SIT Word template. Preserve the stated annotation and translation limitations. Use the reference CSVs for an independent review before presenting the scores as validated accuracy. The archive contains source, data samples, results and figures; weights are downloaded by setup rather than bundled.
