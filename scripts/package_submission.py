"""Package only source and reproducible evidence, excluding weights, environments and caches."""
import hashlib
import json
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for name in ['app.py', 'README.md', '.gitignore', '.gitattributes', 'pytest.ini', 'requirements.txt', 'requirements-lock.txt', 'requirements-report.txt', 'requirements-training.txt']:
        files.append(ROOT / name)
    for name in ['.streamlit', 'policydiff', 'scripts', 'tests', 'data', 'submission', 'notebooks']:
        for path in (ROOT / name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.zip'} and path.name != 'package_manifest.json':
                files.append(path)
    credential_pattern = re.compile(rb'AIza[0-9A-Za-z_-]{35}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    for path in files:
        if credential_pattern.search(path.read_bytes()):
            raise ValueError(f'Potential credential in package input: {path.relative_to(ROOT)}')
    entries = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    manifest = ROOT / 'submission/package_manifest.json'
    manifest.write_text(json.dumps({'sha256': entries, 'excluded': ['.venv', '.cache', 'outputs', 'full PrivaSeer demo', 'the archive itself'],
                                    'annotation_authority': 'AI-reviewed references; independent review pending'}, indent=2), encoding='utf-8')
    files.append(manifest)
    target = ROOT / 'submission/PolicyDiff_Submission.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, Path('PolicyDiff') / path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert all('.cache' not in p.split('/') and '.venv' not in p.split('/') for p in archive.namelist())
    print(f'Packaged {len(files)} files, {target.stat().st_size / 1024 / 1024:.2f} MB: {target}')


if __name__ == '__main__':
    main()
