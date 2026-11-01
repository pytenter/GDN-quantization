#!/usr/bin/env python3
"""CPU-only comparison of actual retained data. Never dispatches model or GPU work."""
import argparse, csv, hashlib, json, math, os
from pathlib import Path, PurePosixPath
GDN_LAYERS=(0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30)
INTERMEDIATE_NAMES=['query','key','value','gate','beta','core_actual','state_actual','native_norm_query','native_norm_key']

def require(ok, message):
    if not ok: raise RuntimeError(message)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(path.read_bytes())

def new_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def safe_file(root, name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts,'Unsafe raw path')
    out=root/p
    require(out.is_file() and not out.is_symlink(),'Missing retained raw file')
    part=root
    for component in p.parts:
        part=part/component;require(not part.is_symlink(),'Raw ancestor symlink prohibited')
    require(root.resolve() in out.resolve().parents,'Raw path escapes root')
    return out

def tensor_bytes(stream,row,torch):
    dtype=getattr(torch,row['dtype'].removeprefix('torch.'))
    count=math.prod(row['shape']);size=torch.empty((),dtype=dtype).element_size()
    require(row['offset']>=0 and row['bytes']==count*size,'Tensor bounds/layout differs')
    stream.seek(row['offset']);body=stream.read(row['bytes'])
    require(len(body)==row['bytes'] and hashlib.sha256(body).hexdigest()==row['sha256'],'Tensor raw SHA/bytes differ')
    tensor=torch.frombuffer(bytearray(body),dtype=dtype).reshape(row['shape']) if count else torch.empty(row['shape'],dtype=dtype)
    require(bool(torch.isfinite(tensor).all()),'Nonfinite retained tensor')
    return body,tensor

def coordinates(name,shape,element,explicit_layer=None):
    import re
    index=None
    if element is not None:
        index=[];n=element
        for d in reversed(shape):index.append(n%d);n//=d
        index=list(reversed(index))
    layer=explicit_layer
    if layer is None:
        for pattern in [r'/layers/(\d+)(?:/|$)',r'/(?:recurrent_states|conv_states)/int:(\d+)(?:/|$)']:
            match=re.search(pattern,name)
            if match:layer=int(match.group(1));break
    head=None
    if index is not None:
        if name=='state_actual' or ('recurrent_states' in name) or (shape==[1,32,128,128]): head=index[1]
        elif name in ['query','key','value','core_actual','native_norm_query','native_norm_key'] and len(shape)==4 and shape[2]==32:head=index[2]
        elif name in ['gate','beta'] and len(shape)==3 and shape[2]==32:head=index[2]
    return dict(layer=layer,head=head,multidimensional_index=index)

def compare_tensor(a,b,ba,bb,torch,token,layer,group):
    layout_equal=all(a[k]==b[k] for k in ['name','dtype','shape','stride','bytes'])
    raw_equal=ba==bb;scalar_equal=bool(torch.equal(a['_tensor'],b['_tensor']))
    if not layout_equal:
        return dict(group=group,token=token,tensor=a['name'],dtype=a['dtype'],shape=a['shape'],
                    actual_dtype=b['dtype'],actual_shape=b['shape'],layout_equal=False,
                    torch_equal=scalar_equal,byte_equal=raw_equal,max_absolute=None,max_relative=None,
                    first_different_byte=None,layer=layer,head=None,multidimensional_index=None,
                    comparison_scope='LAYOUT_MISMATCH_NOT_NUMERICALLY_COMPARABLE')
    x=a['_tensor'].double().reshape(-1);y=b['_tensor'].double().reshape(-1)
    error=(x-y).abs();relative=error/x.abs().clamp_min(1e-300)
    maxabs=float(error.max()) if x.numel() else 0.;maxrel=float(relative.max()) if x.numel() else 0.
    byte_index=None if raw_equal else next(i for i,(u,v) in enumerate(zip(ba,bb)) if u!=v)
    element=None if byte_index is None else byte_index//a['_tensor'].element_size()
    return dict(group=group,token=token,tensor=a['name'],dtype=a['dtype'],shape=a['shape'],
                actual_dtype=b['dtype'],actual_shape=b['shape'],layout_equal=True,
                comparison_scope='VERIFIED_CONTIGUOUS_VALUES_MATCHED_LAYOUT',torch_equal=scalar_equal,byte_equal=raw_equal,max_absolute=maxabs,
                max_relative=maxrel if math.isfinite(maxrel) else str(maxrel),first_different_byte=byte_index,
                **coordinates(a['name'],a['shape'],element,layer))

def compare_records(root,ra,rb,pa,pb,torch,token,layer,group):
    require([r['name'] for r in ra]==[r['name'] for r in rb],'Paired tensor roster differs')
    output=[]
    with pa.open('rb') as fa,pb.open('rb') as fb:
        for a,b in zip(ra,rb):
            ba,ta=tensor_bytes(fa,a,torch);bb,tb=tensor_bytes(fb,b,torch)
            output.append(compare_tensor(dict(a,_tensor=ta),dict(b,_tensor=tb),ba,bb,torch,token,layer,group))
    return output

def token_records(unit):
    rows={read(p)['token']:read(p) for p in sorted(unit.glob('TOKEN_*.json'))}
    divergence=unit/'EARLIEST_DIVERGENCE.json'
    if divergence.exists():
        obj=read(divergence);rows.setdefault(obj['token'],obj['actual'])
    return rows

def reduce(root,output):
    require(not output.exists(),'Reduction output exists; no overwrite')
    import torch
    require(not torch.cuda.is_initialized(),'CPU reducer must not have CUDA context')
    aroot=root/'execution/NUMERIC_FRESH0';broot=root/'execution/NUMERIC_FRESH1'
    aa=token_records(aroot);bb=token_records(broot);pairs=sorted(set(aa)&set(bb));rows=[];metadata=[]
    for token in pairs:
        a,b=aa[token],bb[token]
        for rec in [a,b]:require(rec['token']==token and rec['cursor']==token+1,'Token/cursor identity mismatch')
        require(a['input_id']==b['input_id'],'Input token mismatch')
        pa=aroot/'raw'/('token_%03d.bin'%token);pb=broot/'raw'/pa.name
        for path,rec in [(pa,a),(pb,b)]:require(path.stat().st_size==rec['raw_bytes'] and digest(path)==rec['raw_sha256'],'Token raw container SHA/bytes differ')
        metadata.append(dict(token=token,equal=all(a.get(k)==b.get(k) for k in ['cache_metadata','tensor_storage_aliases','all8attention_KV_lengths'])))
        rows+=compare_records(root,a['tensors'],b['tensors'],pa,pb,torch,token,None,'MODEL_CACHE_LOGITS')
    intermediate_keys=set()
    for pa in sorted((aroot/'intermediates').glob('TOKEN_*_LAYER_*.json')):
        pb=broot/'intermediates'/pa.name
        if not pb.exists():continue
        a,b=read(pa),read(pb)
        require(all(a[k]==b[k] for k in ['token','layer','kind','serialization']),'Intermediate identity differs')
        key=(a['token'],a['layer']);require(key not in intermediate_keys,'Duplicate intermediate token/layer')
        require(key[0] in range(32) and key[1] in GDN_LAYERS,'Unexpected intermediate token/layer')
        require(pa.name=='TOKEN_%03d_LAYER_%02d.json'%key,'Intermediate filename/identity differs')
        require(a['kind']==('chunk' if key[0]==0 else 'recurrent'),'Unexpected native helper kind')
        for rec,unit in [(a,'NUMERIC_FRESH0'),(b,'NUMERIC_FRESH1')]:
            require(rec['unit']==unit,'Intermediate unit identity differs')
            require([x['name'] for x in rec['tensors']]==INTERMEDIATE_NAMES,'Incomplete/changed intermediate tensor roster')
        af=safe_file(root,a['raw_path']);bf=safe_file(root,b['raw_path'])
        require(aroot.resolve() in af.resolve().parents and broot.resolve() in bf.resolve().parents,'Raw unit binding differs')
        for path,rec in [(af,a),(bf,b)]:require(path.stat().st_size==rec['raw_bytes'] and digest(path)==rec['raw_sha256'],'Intermediate raw SHA/bytes differ')
        rows+=compare_records(root,a['tensors'],b['tensors'],af,bf,torch,a['token'],a['layer'],'REAL_GDN_INTERMEDIATES');intermediate_keys.add(key)
    expected_intermediate_keys={(t,layer) for t in range(32) for layer in GDN_LAYERS}
    complete=pairs==list(range(32)) and intermediate_keys==expected_intermediate_keys
    different=[r for r in rows if not(r['layout_equal'] and r['byte_equal'] and r['torch_equal'])]
    result=dict(status=('PASS_FULL_32TOKEN_RETAINED_CPU_PAIR_COMPARISON' if complete and not different and all(x['equal'] for x in metadata) else 'PARTIAL_OR_FAILED_ACTUAL_RETAINED_PAIR_COMPARISON'),
                actual_token_pairs=len(pairs),expected_token_pairs=32,actual_intermediate_pairs=len(intermediate_keys),
                expected_intermediate_pairs=768,actual_tensor_comparisons=len(rows),complete_pair=complete,
                all_byte_equal=all(r['byte_equal'] for r in rows) if rows else None,
                all_torch_equal=all(r['torch_equal'] for r in rows) if rows else None,
                earliest_difference=min(different,key=lambda r:(r['token'],r['layer'] if r['layer'] is not None else -1,r['tensor'])) if different else None,
                earliest_metadata_difference=next((r for r in metadata if not r['equal']),None),
                metadata=metadata,torch_equal_scope='Executed on CPU tensors reconstructed from verified retained contiguous raw; not live GPU tensors',
                GPU_dispatches=0,CUDA_context_initialized=bool(torch.cuda.is_initialized()))
    output.mkdir(parents=True);new_json(output/'FRESH_PROCESS_PARITY.json',result)
    if rows:
        fields=['group','token','layer','head','tensor','dtype','shape','actual_dtype','actual_shape','layout_equal','comparison_scope','torch_equal','byte_equal','max_absolute','max_relative','first_different_byte','multidimensional_index']
        with (output/'ACTUAL_RETAINED_TENSOR_COMPARISONS.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    require(not torch.cuda.is_initialized(),'Reducer unexpectedly initialized CUDA')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='';print(json.dumps(reduce(args.root.resolve(),args.output.resolve()),allow_nan=False))
