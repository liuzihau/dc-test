"""Validate the four raw puzzle files, independently of training or pickle loading."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'.cache/downloads/reasoning-puzzles/Reasoning puzzles public data'
FILES=('Sudoku-train-data.npy','Sudoku-test-data.npy','zebra-train-data.pkl','zebra-test-data.pkl')

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(directory,manifest,check_hashes=False):
    records=[]
    for expected in manifest['files']:
        path=directory/expected['name']
        if not path.is_file():raise FileNotFoundError('Copy the missing puzzle data: '+str(path))
        if path.stat().st_size!=expected['bytes']:raise ValueError('Data size differs: '+str(path))
        if check_hashes and checksum(path)!=expected['sha256']:raise ValueError('Data checksum differs: '+str(path))
        records.append(str(path))
    return records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--write-manifest',type=Path)
    parser.add_argument('--sha256',action='store_true')
    args=parser.parse_args()
    if args.write_manifest:
        manifest=dict(format='original author puzzle files',files=[dict(name=name,bytes=(args.directory/name).stat().st_size,
            sha256=checksum(args.directory/name)) for name in FILES])
        cached=[]
        for directory in ('author-sudoku','author-zebra'):
            for path in sorted((ROOT/'.cache'/directory).glob('*.npz')):
                cached.append(dict(path=str(path.relative_to(ROOT/'.cache')),bytes=path.stat().st_size,sha256=checksum(path)))
        manifest['cached_files']=cached
        args.write_manifest.write_text(json.dumps(manifest,indent=2)+'\n')
        print('Wrote data manifest:',args.write_manifest)
    else:
        manifest=json.loads((ROOT/'puzzle_recurrence/data_manifest.json').read_text())
        paths=verify(args.directory,manifest,args.sha256)
        for expected in manifest.get('cached_files',[]):
            path=ROOT/'.cache'/expected['path']
            if not path.is_file() or path.stat().st_size!=expected['bytes']:raise ValueError('Copy the missing or incomplete prepared cache: '+str(path))
            if args.sha256 and checksum(path)!=expected['sha256']:raise ValueError('Cache checksum differs: '+str(path))
        print('Verified four data files'+(' and SHA256 hashes' if args.sha256 else ' and sizes')+'.')
        for path in paths:print(path)

if __name__=='__main__':main()
