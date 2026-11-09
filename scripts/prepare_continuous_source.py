"""Fetch pinned official StreamVGGT source into a new external runtime folder."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    commit='05682bcb05683c4c71522e9cb618fc8ca349063c'
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    tree=json.load(urlopen('https://api.github.com/repos/wzzheng/StreamVGGT/git/trees/'+commit+'?recursive=1'))
    paths=[r['path'] for r in tree['tree'] if r['type']=='blob' and
           ((r['path'].startswith('src/streamvggt/') and r['path'].endswith('.py'))
            or r['path'].lower() in ['license','license.txt','license.md','readme.md'])]
    def fetch(path):
        with urlopen('https://raw.githubusercontent.com/wzzheng/StreamVGGT/'+commit+'/'+path,timeout=60) as response:
            data=response.read()
        target=root/'upstream'/path
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as out:out.write(data)
        return dict(path=path,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,paths))
    with (root/'source_manifest.json').open('x') as out:
        json.dump(dict(repository='https://github.com/wzzheng/StreamVGGT',commit=commit,files=rows),out,indent=2)
    print(root)


if __name__=='__main__':main()
