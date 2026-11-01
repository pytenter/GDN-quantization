import argparse,hashlib,json,pathlib,time
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.root)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=root/'hashes/artifact_sha256.txt';assert sha(m)=='d659b34a8d8f7dc6196192bcad9d2f95a0aad3d9b1237f58f2e501777c58b015';lines=m.read_text().splitlines();assert len(lines)==70
for line in lines:
 h,rel=line.split('  ',1);assert sha(root/rel)==h,rel
r=dict(status='PASS_UNCHANGED',root=str(root),verified_files=len(lines),manifest_sha256=sha(m),receipt_sha256=sha(root/'hashes/DELIVERY_VERIFICATION.json'),GPU_HOURS_USED=0,torch_imported=False,completed_unix=time.time())
with pathlib.Path(a.output).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
