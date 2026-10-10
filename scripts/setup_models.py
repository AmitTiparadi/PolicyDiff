"""Download pinned models and explicit NLTK resources into the project cache."""
import argparse
import hashlib
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from policydiff.settings import CACHE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hindi", action="store_true")
    args = parser.parse_args()
    import nltk
    import requests
    nltk.data.path.insert(0, str(CACHE / 'nltk'))
    index, records = None, []
    for name in ["stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "cmudict"]:
        category = 'taggers' if 'tagger' in name else 'corpora'
        try:
            nltk.data.find(f'{category}/{name}')
            continue
        except LookupError:
            try:
                nltk.data.find(f'{category}/{name}.zip')
                continue
            except LookupError:
                pass
        if index is None:
            response = requests.get('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml', timeout=60)
            response.raise_for_status()
            index = ET.fromstring(response.content)
        package = index.find(f'.//package[@id="{name}"]')
        if package is None:
            raise ValueError(f'NLTK index has no {name}')
        # Restrict fetched resources to NLTK's publisher repository.
        url = package.attrib['url']
        if not url.startswith('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/'):
            raise ValueError('Unexpected NLTK resource host/path')
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != package.attrib.get('sha256_checksum'):
            raise ValueError(f'NLTK SHA-256 mismatch: {name}')
        target = CACHE / 'nltk' / category
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for member in archive.infolist():
                if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                    raise ValueError('Unsafe archive member path')
            archive.extractall(target)
        records.append({'resource': name, 'url': url, 'sha256': digest})
    if records:
        (CACHE / 'nltk/download_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    from policydiff.models import embedding_model, translation_model
    print("Loading semantic model…", flush=True)
    embedding_model(allow_download=True)
    if not args.skip_hindi:
        print("Loading English–Hindi model…", flush=True)
        translation_model(allow_download=True)
    print("Model setup complete.")


if __name__ == "__main__":
    main()
