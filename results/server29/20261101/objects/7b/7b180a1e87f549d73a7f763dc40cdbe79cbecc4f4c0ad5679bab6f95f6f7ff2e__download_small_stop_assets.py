"""Read-only post-STOP small asset download with per-file remote/local SHA."""
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
meta=json.loads((ROOT/'ROOT.json').read_text());remote=meta['remote_root'];dest=ROOT/'remote_evidence'
if dest.exists():raise RuntimeError('Existing download state; inspect before attempting any supplement')
program=r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(REMOTE)
allowed={'.py','.json','.jsonl','.md','.txt','.csv','.tsv','.log','.sh','.h','.hpp','.c','.cc','.cpp','.cu','.cuh','.toml','.cfg','.ini','.yaml','.yml','.jinja'}
rows=[];skipped=[]
def identity(path):
 before=path.stat();raw=path.read_bytes();after=path.stat()
 if(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('Changed small file:'+str(path))
 return raw,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for path in sorted(root.rglob('*')):
 if not path.is_file():continue
 relative=path.relative_to(root)
 if relative.parts[0]=='independent_delivery':continue
 if path.is_symlink():raise RuntimeError('Unexpected symlink in post-STOP root:'+str(path))
 if path.suffix.lower()not in allowed or path.stat().st_size>16*1024**2:
  skipped.append(dict(path=str(path),reason='NOT_ALLOWED_SMALL_TEXT_SOURCE_OR_METADATA',bytes=path.stat().st_size));continue
 raw,proof=identity(path);rows.append(dict(path='run/'+str(relative),source_remote_path=str(path),kind='FROZEN_RUN_SMALL_ASSET',**proof))
source=json.loads((root/'gpu0/SOURCE_MANIFEST.json').read_text());retained=[]
for row in source['external_dependencies']:
 path=Path(row['path'])
 if '/modelscope_models/'in str(path)or path.suffix.lower()not in allowed or row['bytes']>16*1024**2:
  retained.append(dict(**row,download_status='RETAINED_REMOTE_ONLY_BINARY_MODEL_OR_LARGE_DEPENDENCY'));continue
 raw,proof=identity(path)
 if proof['bytes']!=row['bytes']or proof['sha256']!=row['sha256']:raise RuntimeError('Frozen external source changed:'+str(path))
 key=hashlib.sha256(str(path).encode()).hexdigest()[:16]
 rows.append(dict(path='external/'+key+'/'+path.name,source_remote_path=str(path),kind='FROZEN_EXTERNAL_SMALL_SOURCE_OR_METADATA',**proof))
manifest=dict(status='REMOTE_ENUMERATED_SMALL_ASSETS_WITH_CURRENT_SHA_BYTES',remote_root=str(root),files=rows,files_count=len(rows),total_bytes=sum(x['bytes']for x in rows),skipped_run_files=skipped,retained_remote_dependencies=retained,no_model_or_tensor_download=True,includes_C_CPP_CUDA_headers=True)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')as archive:
 raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode();info=tarfile.TarInfo('DOWNLOAD_MANIFEST.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
 for row in rows:
  raw,proof=identity(Path(row['source_remote_path']))
  if proof['bytes']!=row['bytes']or proof['sha256']!=row['sha256']:raise RuntimeError('Asset changed before tar stream')
  info=tarfile.TarInfo(row['path']);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
'''.replace('Path(REMOTE)','Path('+repr(remote)+')')
command=meta['remote_python']+' -B -c '+shlex.quote(program)
archive_path=ROOT/'independent_delivery/SMALL_STOP_ASSETS.tar.gz'
with archive_path.open('xb')as f:
 proc=subprocess.run(['ssh','-S',meta['ssh_control_path'],'-o','BatchMode=yes','-o','ConnectTimeout=15','zypan@202.38.247.55',command],stdout=f,stderr=subprocess.PIPE,timeout=600)
(ROOT/'independent_delivery/SMALL_STOP_ASSETS_STDERR.txt').write_bytes(proc.stderr)
if proc.returncode:raise RuntimeError('Download failed; retained partial tar must be inspected')
dest.mkdir(exist_ok=False)
with tarfile.open(archive_path,'r:gz')as archive:
 for member in archive:
  path=Path(member.name)
  if path.is_absolute()or'..'in path.parts or not member.isfile():raise RuntimeError('Unsafe tar member')
  target=dest/path;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb')as f:f.write(archive.extractfile(member).read())
manifest=json.loads((dest/'DOWNLOAD_MANIFEST.json').read_text())
for row in manifest['files']:
 path=dest/row['path'];raw=path.read_bytes()
 if len(raw)!=row['bytes']or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Downloaded asset mismatch:'+row['path'])
actual={str(x.relative_to(dest))for x in dest.rglob('*')if x.is_file()}
if actual!={x['path']for x in manifest['files']}|{'DOWNLOAD_MANIFEST.json'}:raise RuntimeError('Download file roster mismatch')
out=dict(status='PASS_ALL_DOWNLOADED_SMALL_ASSET_SHA_BYTES',files_verified=manifest['files_count'],bytes_verified=manifest['total_bytes'],download_manifest_sha256=hashlib.sha256((dest/'DOWNLOAD_MANIFEST.json').read_bytes()).hexdigest(),download_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),retained_remote_dependencies=len(manifest['retained_remote_dependencies']),skipped_run_files=manifest['skipped_run_files'],no_model_or_tensor_download=True)
with(ROOT/'independent_delivery/SMALL_ASSET_DOWNLOAD_VERIFICATION.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
