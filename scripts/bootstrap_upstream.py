"""Download only the pinned, hash-verified official files required by this pilot."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    lock=json.loads((ROOT/'configs/upstream.json').read_text())
    dest=ROOT/'vendor/official-jit'
    def fetch(item):
        name,expected=item
        target=dest/name
        if target.is_file():
            if hashlib.sha256(target.read_bytes()).hexdigest()==expected:
                return f'Verified {name}'
            raise RuntimeError(f'Local upstream file changed: {target}. Preserve it before fetching a fresh copy.')
        url=f'https://raw.githubusercontent.com/LTH14/JiT/{lock["revision"]}/{name}'
        with urllib.request.urlopen(url,timeout=120) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError(f'Hash mismatch: {name}')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
        return f'Downloaded {name}'
    with ThreadPoolExecutor(max_workers=4) as pool:
        for message in pool.map(fetch,lock['files'].items()):print(message)
    print(f'Official JiT revision: {lock["revision"]}')

if __name__=='__main__':main()
