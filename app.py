from pathlib import Path
import hashlib
import json
import streamlit as st

from policydiff.analysis import Config, default_config, add_hindi, compare
from policydiff.report import csv_report, highlighted, html_report, json_report
from policydiff.settings import ROOT
from policydiff.text import load_bytes, load_file

st.set_page_config(page_title="PolicyDiff", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container{max-width:1280px;padding-top:2.5rem}
h1{letter-spacing:-.045em} [data-testid="stMetric"]{background:white;border:1px solid #dee6ea;border-radius:12px;padding:16px}
del{background:#ffe0df;color:#8d2923}mark{background:#c9eee4;color:#164f43}
.clause{line-height:1.75;padding:18px;background:white;border:1px solid #dee6ea;border-radius:10px;min-height:90px}
</style>""", unsafe_allow_html=True)
st.caption("POLICYDIFF  /  PRIVACY POLICY ANALYSIS")
st.title("See what changed. Understand what matters.")
st.write("Compare two versions of the same privacy policy, inspect the clause evidence, and read changes in Hindi.")

with st.sidebar:
    st.subheader("Comparison settings")
    engine_label = st.selectbox("Comparison method", ["Semantic · Sentence-BERT", "Lexical · TF-IDF", "Textual · difflib"])
    engine = {"Semantic · Sentence-BERT": "semantic", "Lexical · TF-IDF": "tfidf", "Textual · difflib": "textual"}[engine_label]
    defaults = default_config(engine)
    with st.expander("Advanced thresholds"):
        match = st.slider("Minimum match similarity", 0.1, 0.95, defaults.match_threshold, 0.01, key=f'match_{engine}')
        reword = st.slider("Likely paraphrase similarity", match, 1.0, max(match, defaults.reword_threshold), 0.01, key=f'reword_{engine}')
        st.caption("Semantic defaults selected on Google development pairs. Similarity scores are not confidence probabilities.")
    st.divider()
    st.caption("Files are processed locally. Initial model setup requires downloads; policy text is not sent to a hosted inference service.")
    results_path = ROOT / 'submission/results/benchmark_results.json'
    if results_path.exists():
        with st.expander('Project evaluation'):
            results = json.loads(results_path.read_text(encoding='utf-8'))
            st.caption(results['annotation_authority'])
            st.table([{'Method': method, 'Test macro F1': round(value['aggregate']['test']['macro_f1_active'], 3)}
                      for method, value in results['methods'].items()])
            st.download_button('Download evaluation JSON', results_path.read_bytes(), 'benchmark_results.json', 'application/json')
            pdf = ROOT / 'submission/PolicyDiff_Report.pdf'
            if pdf.exists():
                st.download_button('Download project report', pdf.read_bytes(), pdf.name, 'application/pdf')

mode = st.radio("Policy source", ["Upload two policies", "Try a sample"], horizontal=True)
try:
    if mode == "Upload two policies":
        a, b = st.columns(2)
        old_upload = a.file_uploader("Earlier version", type=["txt", "html", "htm", "md"])
        new_upload = b.file_uploader("Later version", type=["txt", "html", "htm", "md"])
        if old_upload and new_upload:
            old_text, new_text = load_bytes(old_upload.getvalue(), old_upload.name), load_bytes(new_upload.getvalue(), new_upload.name)
            names = (old_upload.name, new_upload.name)
        else:
            old_text = new_text = ""
            names = ("Earlier policy", "Later policy")
    else:
        samples = {"Synthetic · retention, sharing, additions": ROOT / "data" / "examples"}
        for folder in sorted((ROOT / "data" / "historical").glob("*")):
            if folder.is_dir() and (folder / "old.md").exists() and (folder / "new.md").exists():
                samples[f"Historical · {folder.name}"] = folder
        choice = st.selectbox("Sample pair", list(samples))
        folder = samples[choice]
        extension = "txt" if choice.startswith("Synthetic") else "md"
        old_text, new_text = load_file(folder / f"old.{extension}"), load_file(folder / f"new.{extension}")
        names = (f"{folder.name} · earlier", f"{folder.name} · later")
        if extension == "txt":
            st.caption("This sample is deliberately constructed to demonstrate changes; it is not historical policy evidence.")
        else:
            st.caption("Historical policy versions with source dates and provenance recorded beside the policy files.")
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

fingerprint = hashlib.sha256(repr((old_text, new_text, engine, match, reword)).encode()).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state.pop("report", None)
if st.button("Compare policies", type="primary", disabled=not old_text or not new_text):
    try:
        with st.spinner("Aligning clauses and checking changes…"):
            report = compare(old_text, new_text, Config(engine=engine, match_threshold=match, reword_threshold=reword,
                                                       use_critical_checks=engine == 'semantic'))
            report["source_names"] = list(names)
            st.session_state.report = report
            st.session_state.fingerprint = fingerprint
    except Exception as exc:
        st.error(f"Comparison could not finish: {exc}")
        st.info("For missing models, run the setup command in README.md. You can also choose a lexical baseline.")

if "report" not in st.session_state:
    st.info("Upload an earlier and later policy, or choose a sample pair to begin.")
    st.stop()

report = st.session_state.report
columns = st.columns(6)
for column, (label, count) in zip(columns, report["counts"].items()):
    column.metric(label.replace("review", "Needs review").title(), count)
st.caption(f'{report["old_clauses"]} earlier clauses · {report["new_clauses"]} later clauses · {engine_label}')
summary_tab, evidence_tab, readability_tab, download_tab = st.tabs(["Key changes", "Clause evidence", "Readability", "Download report"])

with summary_tab:
    if not report["summary"]:
        st.success("No substantive change candidates detected. Inspect any reworded clauses in the evidence tab.")
    for item in report["summary"]:
        with st.container(border=True):
            st.markdown(f'**{item["status"].title()} · {item["section"]}**')
            if item["signals"]:
                st.caption(" · ".join(item["signals"]))
            a, b = st.columns(2)
            a.caption("BEFORE")
            a.write(item["old"] or "—")
            b.caption("AFTER")
            b.write(item["new"] or "—")
    st.caption("Extractive overview ranked by change signals and TextRank; inspect all clauses for completeness.")

with evidence_tab:
    if st.button("Translate changed clauses into Hindi", disabled="translation_model" in report):
        try:
            with st.spinner("Translating earlier and later changed clauses…"):
                add_hindi(report)
        except Exception as exc:
            st.error(f"Hindi translation could not finish: {exc}")
    if "translation_model" in report:
        st.caption("Hindi is machine translated. Keep the original English text available when reviewing meaning.")
    selected = st.multiselect("Show categories", list(report["counts"]),
                              default=["modified", "added", "removed", "review", "reworded"])
    for row in report["changes"]:
        if row["status"] not in selected:
            continue
        clause = row["new"] or row["old"]
        score = f' · similarity {row["similarity"]:.2f}' if row["similarity"] is not None else ""
        with st.expander(f'{row["status"].upper()} · {clause["section"]} · clause {clause["id"] + 1}{score}', expanded=row["status"] == "modified"):
            st.caption(row["reason"])
            if row["signals"]:
                st.write(" · ".join(row["signals"]))
            left, right = highlighted(row["old"]["text"] if row["old"] else "", row["new"]["text"] if row["new"] else "")
            a, b = st.columns(2)
            for column, side, value in [(a, "old", left), (b, "new", right)]:
                column.caption("BEFORE" if side == "old" else "AFTER")
                column.markdown(f'<div class="clause">{value or "—"}</div>', unsafe_allow_html=True)
                if side in row.get("hindi", {}):
                    column.write(row["hindi"][side])
                if row[side]:
                    column.caption(f'Clause {row[side]["id"] + 1} · cleaned text offsets {row[side]["start"]}–{row[side]["end"]}')

with readability_tab:
    st.table([{ "Measure": key.replace("_", " ").title(), "Earlier": report["readability"]["old"][key],
               "Later": report["readability"]["new"][key], "Difference": report["readability"]["delta"][key]}
              for key in report["readability"]["old"]])
    st.caption("Higher Flesch Reading Ease indicates easier prose; higher grade level indicates harder prose. Scores do not measure privacy protection.")
    st.write("Key terms: " + ", ".join(report["keywords"]))

with download_tab:
    st.download_button("Download JSON", json_report(report), "policydiff.json", "application/json")
    st.download_button("Download CSV", csv_report(report).encode("utf-8"), "policydiff.csv", "text/csv")
    st.download_button("Download readable HTML", html_report(report), "policydiff.html", "text/html")
    for note in report["notes"]:
        st.caption(note)
