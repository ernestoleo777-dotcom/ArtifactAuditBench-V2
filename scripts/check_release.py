"""Verify the explicit public inventory; fail on missing, changed or unexpected assets."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'manifests/public_files.json'
IGNORED_DIRS = {'.git', '.venv', '.pytest_cache', '__pycache__', 'runs', 'matplotlib-cache'}


def public_files():
    return {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
            if p.is_file() and not any(part in IGNORED_DIRS for part in p.relative_to(ROOT).parts)
            and p.name != '.DS_Store' and p.suffix not in {'.pyc', '.pyo'}}


def main():
    expected = json.loads((ROOT/MANIFEST).read_text())['files']
    actual = public_files()
    assert set(expected) == set(actual)-{MANIFEST}, 'public inventory differs'
    credential = re.compile(r'sk' + r'-(?:or-v1-)?[A-Za-z0-9_-]{24,}')
    private_path = re.compile(r'/(?:Users|Volumes)/[^\s"\']+')
    for name, digest in expected.items():
        content=actual[name].read_bytes()
        assert hashlib.sha256(content).hexdigest()==digest, f'hash mismatch: {name}'
        if actual[name].suffix in {'.py','.json','.jsonl','.csv','.md','.txt','.yml','.cff'}:
            text=content.decode('utf-8')
            assert not credential.search(text), f'credential-shaped value: {name}'
            assert not private_path.search(text), f'machine-local path: {name}'
        assert not any(part in {'cache','requests','responses','iclr2027_repair','repair','extensions'}
                       for part in Path(name).parts), f'excluded asset class: {name}'
    for name, path in actual.items():
        if name.endswith('.md'):
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if '://' not in target and target.split('#')[0]:
                    assert (path.parent/target.split('#')[0]).exists(), f'broken link: {name}: {target}'
    print(json.dumps({'status':'PASS','files_verified':len(expected),'scope':'FILE_IDENTITY_AND_PUBLIC_BOUNDARIES_NOT_SCIENTIFIC_VALIDITY'}))


if __name__=='__main__':
    main()
