import csv
from difflib import SequenceMatcher
import html
import io
import json


def json_report(report):
    return json.dumps(report, ensure_ascii=False, indent=2)


def csv_report(report):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["status", "old_id", "new_id", "similarity", "signals", "old_text", "new_text", "old_hindi", "new_hindi"])
    for r in report["changes"]:
        cells = [r["status"], r["old"]["id"] if r["old"] else "", r["new"]["id"] if r["new"] else "",
                         r["similarity"], "; ".join(r["signals"]), r["old"]["text"] if r["old"] else "",
                         r["new"]["text"] if r["new"] else "", r.get("hindi", {}).get("old", ""),
                         r.get("hindi", {}).get("new", "")]
        writer.writerow(["'" + cell if isinstance(cell, str) and cell.lstrip().startswith(("=", "+", "-", "@")) else cell
                         for cell in cells])
    # UTF-8 BOM makes Hindi readable in Windows spreadsheet applications.
    return "\ufeff" + stream.getvalue()


def highlighted(old, new):
    a, b = (old or "").split(), (new or "").split()
    left, right = [], []
    for tag, i, j, x, y in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        original, updated = html.escape(" ".join(a[i:j])), html.escape(" ".join(b[x:y]))
        if original:
            left.append(original if tag == "equal" else f"<del>{original}</del>")
        if updated:
            right.append(updated if tag == "equal" else f"<mark>{updated}</mark>")
    return " ".join(left), " ".join(right)


def html_report(report):
    rows = []
    for r in report["changes"]:
        old, new = highlighted(r["old"]["text"] if r["old"] else "", r["new"]["text"] if r["new"] else "")
        hindi = r.get("hindi", {})
        rows.append(f'<tr><td>{html.escape(r["status"])}<br>{html.escape("; ".join(r["signals"]))}</td>'
                    f'<td>{old}<p lang="hi">{html.escape(hindi.get("old", ""))}</p></td>'
                    f'<td>{new}<p lang="hi">{html.escape(hindi.get("new", ""))}</p></td></tr>')
    summary = "".join(f'<li><b>{html.escape(r["status"])} — {html.escape(r["section"])}</b>'
                      f'<p>Before: {html.escape(r["old"] or "—")}</p><p>After: {html.escape(r["new"] or "—")}</p></li>'
                      for r in report["summary"])
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>PolicyDiff report</title>' + \
        '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172b3a}' + \
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:16px;text-align:left;vertical-align:top}' + \
        'del{background:#ffe0df}mark{background:#c9eee4}p{line-height:1.6}</style>' + \
        f'<h1>PolicyDiff</h1><p>Engine: {html.escape(report["config"]["engine"])}</p>' + \
        f'<p>{html.escape(json.dumps(report["counts"]))}</p><h2>Key changes</h2><ul>{summary}</ul>' + \
        '<h2>Readability</h2><pre>' + html.escape(json.dumps(report["readability"], indent=2)) + '</pre>' + \
        '<h2>Clause evidence</h2><table><tr><th>Change</th><th>Before</th><th>After</th></tr>' + \
        "".join(rows) + '</table><p>' + html.escape(" ".join(report["notes"])) + '</p></html>'
