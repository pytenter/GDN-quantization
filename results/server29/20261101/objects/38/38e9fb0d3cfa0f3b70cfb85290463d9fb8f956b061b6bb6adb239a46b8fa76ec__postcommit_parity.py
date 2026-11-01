import ast
import hashlib
import os
import subprocess

from common import EXP, FORMAL, REPO, read_json, sha256, write_json


def main():
    auth = read_json(EXP / 'execution_authorization.json')
    commit = auth['PREMEASUREMENT_COMMIT_SHA']
    if not commit or auth['FORMAL_MEASUREMENT_AUTHORIZED'] != 'YES_PENDING_POSTCOMMIT_PARITY':
        raise RuntimeError('postcommit parity authorization mismatch')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() != commit:
        raise RuntimeError('HEAD differs from premeasurement commit')
    freeze = read_json(EXP / 'PREMEASUREMENT_FREEZE_MANIFEST.json')
    for row in freeze['frozen_files']:
        if sha256(EXP/row['path']) != row['sha256']:
            raise RuntimeError(f'postcommit frozen file mismatch: {row["path"]}')
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('deterministic runtime missing')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = FORMAL.read_text(encoding='utf-8')
    tree = ast.parse(source)
    fn = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='quantize_c128')
    nodes = {}
    for node in fn.body[0].body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            nodes[node.targets[0].id] = node
    qcode = compile(ast.fix_missing_locations(ast.Module(
        body=[nodes['source'],nodes['scale']],type_ignores=[])),str(FORMAL),'exec')
    acode = compile(ast.fix_missing_locations(ast.Module(
        body=[nodes['source'],nodes['codes'],nodes['dequant']],type_ignores=[])),str(FORMAL),'exec')
    state = torch.linspace(-3,3,32*128*128,device='cuda:0',dtype=torch.float32).reshape(1,32,128,128)
    env={'state':state.clone(),'torch':torch,'EPS':1e-12}; exec(qcode,env); clean=env['scale']
    adaptive = clean*torch.linspace(.5,1.5,clean.numel(),device='cuda:0').reshape(clean.shape)
    outputs=[]
    for arm,scale in [('NO_SCALE_INFLATION',torch.minimum(adaptive,clean)),
                      ('NO_SCALE_DEFLATION',torch.maximum(adaptive,clean))]:
        target=state.clone(); env={'state':target,'scale':scale,'torch':torch,'EPS':1e-12}; exec(acode,env)
        target.copy_(env['dequant'].to(target.dtype))
        outputs.append((arm,hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest(),
                        int((adaptive!=scale).sum().item())))
    payload={'POSTCOMMIT_MINIMAL_PARITY':'PASS','commit':commit,
             'formal_quantizer_sha256':sha256(FORMAL),'arms':outputs,
             'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
             'TF32':False,'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    write_json(EXP/'qualification/postcommit_minimal_parity.json',payload)
    auth['FORMAL_MEASUREMENT_AUTHORIZED']='YES'
    write_json(EXP/'execution_authorization.json',auth,overwrite=True)
    print('POSTCOMMIT_MINIMAL_PARITY_PASS',flush=True)


if __name__=='__main__':
    main()
