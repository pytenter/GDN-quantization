"""Fresh CPU stream hashes of every frozen model/tokenizer file; no model load."""
import argparse,hashlib,json,pathlib,sys,time,traceback

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def stat_id(p):
    s=pathlib.Path(p).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)

def main(a):
    start=time.monotonic();expected=pathlib.Path(a.expected);rows=[];failure=None
    try:
        source=json.loads(expected.read_text());assert source['status']=='PASS'
        entries=source['model_weight_and_config_current_content_hashes'];assert len(entries)==14
        for x in entries:
            p=pathlib.Path(x['path']);before=stat_id(p);actual=sha(p);after=stat_id(p)
            assert before==after,('file changed during full hash',str(p))
            assert before['bytes']==x['bytes'] and actual==x['sha256'],('content mismatch',str(p),actual)
            rows.append(dict(path=str(p),bytes=before['bytes'],sha256=actual,status='PASS',
                             fresh_full_stream_hash=True,unchanged_during_hash=True,stat=after))
            print(json.dumps(dict(verified=len(rows),files=len(entries),path=str(p))),flush=True)
    except Exception as e:failure=dict(error=repr(e),traceback=traceback.format_exc())
    result=dict(status='FAIL' if failure else 'PASS',stage='NEW_DIRECT_32K_G0_FRESH_CURRENT_CONTENT',
                expected_source=str(expected),expected_source_sha256=sha(expected),
                model_weight_and_config_current_content_hashes=rows,files_verified=len(rows),
                total_bytes=sum(x['bytes'] for x in rows),CPU_wall_seconds=time.monotonic()-start,
                GPU_HOURS_USED=0,torch_imported=False,CUDA_MODEL_LOAD=False,model_forward=False,
                first_failure=failure,script_sha256=sha(__file__),verified_unix=time.time())
    with pathlib.Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],files_verified=len(rows),total_bytes=result['total_bytes'],CPU_wall_seconds=result['CPU_wall_seconds'])))
    if failure:sys.exit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expected',required=True);p.add_argument('--output',required=True);main(p.parse_args())
