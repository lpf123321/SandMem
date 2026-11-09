"""Download immutable VSI-Bench videos, verify LFS digests, then extract."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(root, revision, item):
    target = root / item['path']
    expected = item.get('lfs', {}).get('oid')
    if expected is None:
        raise ValueError('LFS checksum missing: ' + item['path'])
    part = target.with_suffix(target.suffix + '.part')
    if not target.exists():
        for attempt in range(5):
            offset = part.stat().st_size if part.exists() else 0
            if offset == item['size']:
                break
            try:
                url = (f'https://huggingface.co/datasets/nyu-visionx/VSI-Bench/'
                       f'resolve/{revision}/{item["path"]}?download=true')
                with requests.get(url, headers={'Range': f'bytes={offset}-'},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        raise RuntimeError('server did not honor resume Range')
                    with part.open('ab') as output:
                        for block in response.iter_content(4 * 1024 * 1024):
                            output.write(block)
                if part.stat().st_size != item['size']:
                    raise RuntimeError('incomplete archive')
                break
            except Exception as error:
                print(item['path'], 'attempt', attempt + 1, str(error), flush=True)
                if attempt == 4:
                    raise
                time.sleep(3)
        digest = sha256(part)
        if digest != expected:
            raise RuntimeError('archive SHA256 mismatch: ' + item['path'])
        part.rename(target)
    else:
        digest = sha256(target)
        if digest != expected:
            raise RuntimeError('existing archive SHA256 mismatch')
    videos = root / 'videos'
    videos.mkdir(exist_ok=True)
    members = []
    with zipfile.ZipFile(target) as archive:
        for member in archive.infolist():
            destination = (videos / member.filename).resolve()
            if videos.resolve() not in destination.parents:
                raise ValueError('unsafe archive member')
            if member.is_dir():
                continue
            if not destination.exists():
                archive.extract(member, videos)
            if destination.stat().st_size != member.file_size:
                raise RuntimeError('extracted size mismatch')
            members.append(dict(path=member.filename, size=member.file_size, crc32=member.CRC))
    receipt = dict(archive=item['path'], sha256=digest, members=members,
                   revision=revision, status='verified_and_extracted')
    (root / (item['path'] + '.receipt.json')).write_text(json.dumps(receipt, indent=2))
    print(item['path'], 'verified and extracted', len(members), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    revision = json.loads((args.root / 'sources/vsibench_hf_api.json').read_text())['sha']
    tree = json.loads((args.root / 'sources/dataset_tree.json').read_text())
    archives = [entry for entry in tree if entry['path'].endswith('.zip')]
    with ThreadPoolExecutor(3) as pool:
        receipts = list(pool.map(lambda item: download(args.root / 'data', revision, item), archives))
    (args.root / 'reports/video_download.json').write_text(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
