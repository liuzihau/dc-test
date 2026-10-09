"""Download the official OWT full-block reference at a pinned Hub revision."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

REPO = 'kuleshov-group/bd3lm-owt-block_size1024-pretrain'
FILES = ('config.json', 'configuration_bd3lm.py', 'modeling_bd3lm.py', 'model.safetensors')
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/pretrained/bd3lm-owt-block_size1024-pretrain')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / 'hub_metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        response = requests.get(f'https://huggingface.co/api/models/{REPO}?blobs=true', timeout=30)
        response.raise_for_status()
        metadata = response.json()
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    revision = metadata['sha']
    directory = args.output / revision
    directory.mkdir(exist_ok=True)
    remote = {item['rfilename']: item for item in metadata['siblings']}
    records = {}
    for filename in FILES:
        if args.metadata_only and filename == 'model.safetensors':
            continue
        path = directory / filename
        expected = remote[filename].get('lfs', {}).get('sha256')
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.partial')
            with requests.get(f'https://huggingface.co/{REPO}/resolve/{revision}/{filename}',
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                written = 0
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        stream.write(chunk)
                        written += len(chunk)
                        if written // (64 * 1024 * 1024) != (written - len(chunk)) // (64 * 1024 * 1024):
                            print(filename, written // (1024 * 1024), 'MiB', flush=True)
            if expected and sha256(partial) != expected:
                raise ValueError('Hub weight checksum mismatch')
            partial.replace(path)
        digest = sha256(path)
        if expected and digest != expected:
            raise ValueError('Saved weight checksum mismatch')
        records[filename] = {'sha256': digest, 'bytes': path.stat().st_size}
        print('Verified', filename, flush=True)
    (args.output / ('metadata_receipt.json' if args.metadata_only else 'download_receipt.json')).write_text(
        json.dumps({'repo': REPO, 'revision': revision, 'directory': str(directory.resolve()),
                    'files': records, 'weights_downloaded': 'model.safetensors' in records}, indent=2) + '\n')
    print('Reference directory:', directory, flush=True)


if __name__ == '__main__':
    main()
