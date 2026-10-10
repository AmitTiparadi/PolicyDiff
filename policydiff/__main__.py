import argparse
from pathlib import Path
from .analysis import default_config, add_hindi, compare
from .report import html_report, json_report
from .text import load_file


def main():
    parser = argparse.ArgumentParser(description="Compare two versions of a privacy policy.")
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--engine", choices=["semantic", "tfidf", "textual"], default="semantic")
    parser.add_argument("--hindi", action="store_true")
    parser.add_argument("--output", default="outputs/comparison.json")
    args = parser.parse_args()
    report = compare(load_file(args.old), load_file(args.new), default_config(args.engine))
    if args.hindi:
        add_hindi(report)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json_report(report), encoding="utf-8")
    target.with_suffix(".html").write_text(html_report(report), encoding="utf-8")
    print(report["counts"])
    print(f"Report: {target.resolve()}")


if __name__ == "__main__":
    main()
