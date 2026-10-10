"""Source small, pinned publisher policy histories with provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from policydiff.settings import ROOT

PAIRS = [('20240328', '20240916'), ('20230701', '20231004'), ('20190122', '20191015')]
REPOS = {
    'mozilla': ('mozilla/legal-docs', 'en/websites_privacy_notice.md'),
    'automattic': ('Automattic/legalmattic', 'Privacy-Policy.md'),
}


def sanitize_html(data):
    """Redact embedded Google API keys without changing policy paragraphs."""
    return re.sub(rb'AIza[0-9A-Za-z_-]{35}', b'REDACTED_GOOGLE_API_KEY', data)


def verify_cached(folder):
    record = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    for side in ['old', 'new']:
        expected = record[side].get('text_sha256') or record[side]['sha256']
        if hashlib.sha256((folder / f'{side}.md').read_bytes()).hexdigest() != expected:
            raise ValueError(f'Cached source hash mismatch: {folder.name}/{side}.md')
        if (folder / f'{side}.html').exists():
            if hashlib.sha256((folder / f'{side}.html').read_bytes()).hexdigest() != record[side].get('stored_html_sha256', record[side]['raw_sha256']):
                raise ValueError(f'Cached HTML hash mismatch: {folder.name}/{side}.html')


def source_repository(publisher, output, refresh=False):
    cached = sorted(output.glob(f'{publisher}-*'))
    if not refresh and len(cached) == 2 and all((f / 'provenance.json').exists() for f in cached):
        for folder in cached:
            verify_cached(folder)
        print(f'{publisher}: reused two verified publisher pairs', flush=True)
        return
    repo, path = REPOS[publisher]
    session = requests.Session()
    session.headers['User-Agent'] = 'PolicyDiff-academic-sample/2.0'
    versions = []
    for year in [2022, 2023, 2024]:
        response = session.get(f'https://api.github.com/repos/{repo}/commits',
                               params={'path': path, 'until': f'{year}-12-31T23:59:59Z', 'per_page': 1}, timeout=60)
        response.raise_for_status()
        commits = response.json()
        if not commits:
            raise ValueError(f'No {publisher} snapshot before end of {year}')
        commit = commits[0]
        url = f'https://raw.githubusercontent.com/{repo}/{commit["sha"]}/{path}'
        raw = session.get(url, timeout=60)
        raw.raise_for_status()
        versions.append({'content': raw.content, 'sha': commit['sha'], 'source_url': url,
                         'commit_url': commit['html_url'], 'snapshot_date': commit['commit']['author']['date'],
                         'sha256': hashlib.sha256(raw.content).hexdigest()})
    for before, after in zip(versions, versions[1:]):
        if before['sha256'] == after['sha256']:
            raise ValueError(f'{publisher} snapshots contain identical text')
        folder = output / f'{publisher}-{before["snapshot_date"][:10]}-{after["snapshot_date"][:10]}'
        folder.mkdir(parents=True, exist_ok=True)
        provenance = {'publisher': publisher, 'domain': 'mozilla.org' if publisher == 'mozilla' else 'automattic.com',
                      'dataset': 'Publisher-maintained policy Git history', 'repository': f'https://github.com/{repo}',
                      'repository_path': path, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                      'annotation_status': 'Unannotated real historical pair',
                      'selection': 'Last file commit before end of 2022, 2023, 2024; adjacent distinct versions',
                      'date_note': 'Snapshot date is a Git author timestamp, not necessarily policy effective date.'}
        for label, snapshot in [('old', before), ('new', after)]:
            (folder / f'{label}.md').write_bytes(snapshot['content'])
            provenance[label] = {k: v for k, v in snapshot.items() if k != 'content'}
        (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        print(f'{publisher}: {before["snapshot_date"]} -> {after["snapshot_date"]}', flush=True)


def extract_policy(data):
    soup = BeautifulSoup(data, 'html.parser')
    root = soup.find('main') or soup.find(attrs={'role': 'main'})
    if root is None:
        raise ValueError('Publisher layout changed: no main policy container found.')
    for node in root.find_all(['script', 'style', 'nav', 'footer', 'aside', 'dialog']):
        node.decompose()
    lines = []
    blocks = ['h1', 'h2', 'h3', 'h4', 'p', 'li']
    for node in root.find_all(blocks):
        if node.find(blocks):
            continue
        text = ' '.join(node.get_text(' ', strip=True).split())
        if text:
            lines.append(('## ' if node.name.startswith('h') else '') + text)
    result = '\n\n'.join(lines)
    if len(result) < 10000 or 'Google Privacy Policy' not in result:
        raise ValueError('Publisher response does not contain the expected English policy.')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'historical')
    parser.add_argument('--publishers', nargs='+', choices=['google', 'mozilla', 'automattic'], default=['google', 'mozilla', 'automattic'])
    parser.add_argument('--refresh', action='store_true', help='Replace cached sources. Review/regenerate annotations if source hashes change.')
    args = parser.parse_args()
    session = requests.Session()
    session.headers['User-Agent'] = 'PolicyDiff-academic-sample/1.0'
    for before, after in PAIRS if 'google' in args.publishers else []:
        folder = args.output / f'google-{before}-{after}'
        if not args.refresh and (folder / 'provenance.json').exists():
            verify_cached(folder)
            print(f'Google: reused verified {before} -> {after}', flush=True)
            continue
        versions = []
        for label, date in [('old', before), ('new', after)]:
            url = f'https://policies.google.com/privacy/archive/{date}?hl=en'
            response = session.get(url, timeout=60)
            response.raise_for_status()
            text = extract_policy(response.content)
            expected = datetime.strptime(date, '%Y%m%d')
            effective = f'{expected.strftime("%B")} {expected.day}, {expected.year}'
            if f'Effective {effective}' not in text:
                raise ValueError(f'Expected policy effective date {effective} was not found.')
            versions.append((label, date, response, text))
        if versions[0][3] == versions[1][3]:
            raise ValueError('Archive versions unexpectedly contain identical text.')
        folder.mkdir(parents=True, exist_ok=True)
        provenance = {'domain': 'google.com', 'dataset': 'Google publisher privacy policy archive',
                      'archive_index': 'https://policies.google.com/privacy/archive',
                      'retrieved_at': datetime.now(timezone.utc).isoformat(),
                      'annotation_status': 'Unannotated real policy pair; not part of PrivaSeer or Princeton corpus.',
                      'extraction': 'Leaf headings, paragraphs, and list items in the main HTML container; HTML retained with embedded Google API keys redacted.',
                      'rights': 'Original policy text belongs to the publisher; retained for academic comparison.'}
        for label, date, response, text in versions:
            stored_html = sanitize_html(response.content)
            (folder / f'{label}.html').write_bytes(stored_html)
            (folder / f'{label}.md').write_bytes(text.encode('utf-8'))
            provenance[label] = {'effective_date': datetime.strptime(date, '%Y%m%d').date().isoformat(),
                                 'source_url': response.url, 'raw_sha256': hashlib.sha256(response.content).hexdigest(),
                                 'stored_html_sha256': hashlib.sha256(stored_html).hexdigest(),
                                 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}
        (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        print(f'Google: {before} -> {after}', flush=True)
    for publisher in args.publishers:
        if publisher in REPOS:
            source_repository(publisher, args.output, args.refresh)


if __name__ == '__main__':
    main()
