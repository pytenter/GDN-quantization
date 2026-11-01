"""CPU-only synthetic control tests; no model imports or scientific execution."""
import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import amended_canonical3090_pipeline as P
import amended_c5r_reproduction_gate as G

def write(p,r):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(r,f)

class ControlTests(unittest.TestCase):
    def fixture(self,root):
        write(root/'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json',{'original_preregistration_rewritten':False})
        write(root/'training/C5R/summary.json',{'status':'PASS','updates':125,'exposures':1000})
        write(root/'training/C5R/checkpoint_selection.json',{'selected':{'update':63}})
        paths=('cross_objective_learning_curves.csv','shared16_comparison.csv','rotation_distances_descriptive.csv')
        for name in paths:
            with (root/'analysis'/name).open('x',newline='') as f:
                if name==paths[0]:
                    w=csv.DictWriter(f,fieldnames=['condition','update','exposure','state','functional','final_hidden']);w.writeheader()
                    for c,n in [('C5R',21),('historical_C5',20)]:
                        for i in range(n):w.writerow(dict(condition=c,update=i,exposure=i*8,state=.1/(i+1),functional=.2/(i+1),final_hidden='' if c.startswith('historical') else .3))
                elif name==paths[1]:
                    w=csv.DictWriter(f,fieldnames=['condition','state','functional','final_hidden']);w.writeheader()
                    for c in ('C5R','historical_C5_u063'):w.writerow(dict(condition=c,state=.01,functional=.04,final_hidden=.05))
                else:f.write('candidate,reference,layer,delta_frobenius_distance\n')
        write(root/'analysis/shared16_capture_metrics.json',{})
        (root/'analysis/historical_reproduction_analysis.md').write_text('Synthetic non-AIME evidence')
        (root/'aime26').mkdir()

    def test_routes(self):
        for category,route,arms in [('CLOSE_REPRODUCTION','SKIP_NOT_NEEDED',('C6R','C7')),('MATERIAL_SHIFT','RUN_DUE_TO_MATERIAL_SHIFT',('C6R','C7','C5R')),('INCONCLUSIVE','SKIP_INCONCLUSIVE',('C6R','C7')),('NOT_COMPLETED','NOT_REACHED',('C6R','C7'))]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);a=root/'protocol/PROTOCOL_AMENDMENT_C5R_AUXILIARY_ENDPOINT_V1.json';write(a,{})
                write(root/'analysis/C5R_REPRODUCTION_GATE.json',{'status':'PASS','no_AIME_used':True,'amendment_sha256':P.sha(a),'C5R_REPRODUCTION':category,'C5R_AIME26_ROUTE':route,'evidence_sha256':{}})
                with patch.object(P,'ROOT',root):self.assertEqual(P.active_arms(),arms)
                gate=P.read(root/'analysis/C5R_REPRODUCTION_GATE.json');gate['C5R_AIME26_ROUTE']='UNAUTHORIZED'
                (root/'analysis/C5R_REPRODUCTION_GATE.json').write_text(json.dumps(gate))
                with patch.object(P,'ROOT',root):
                    with self.assertRaises(RuntimeError):P.active_arms()

    def test_reproduction_default_and_no_AIME(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'analysis').mkdir();self.fixture(root)
            with patch.object(G,'ROOT',root):G.main()
            r=P.read(root/'analysis/C5R_REPRODUCTION_GATE.json')
            self.assertEqual(r['C5R_AIME26_ROUTE'],'SKIP_INCONCLUSIVE')
            self.assertTrue((root/'aime26/C5R_NOT_RUN.json').exists())
            self.assertFalse(list((root/'aime26').glob('*.csv')))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'analysis').mkdir();self.fixture(root)
            (root/'aime26/forbidden.jsonl').write_text('{}\n')
            with patch.object(G,'ROOT',root):
                with self.assertRaises(RuntimeError):G.main()

    def test_static_family(self):
        self.assertEqual(len(P.MANDATORY_PAIRS),6)
        self.assertEqual(len(P.AUXILIARY_PAIRS),2)
        self.assertEqual(P.MANDATORY_PAIRS[0],('C7','C6R'))
        self.assertFalse(any('C5R' in pair for pair in P.MANDATORY_PAIRS))
        self.assertEqual(P.ALL_ARMS.index('C6R')*60,60)
        self.assertEqual(P.ALL_ARMS.index('C7')*60,120)

    def test_final_verdicts_and_skipped_auxiliary(self):
        for net,hnet,label in [(1,1,'C7_OBJECTIVE_SUPPORTED_AND_BEATS_HADAMARD'),(1,0,'C7_OBJECTIVE_SUPPORTED_BUT_BELOW_HADAMARD'),(0,-2,'C7_OBJECTIVE_NOT_SUPPORTED')]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'analysis').mkdir()
                write(root/'provenance/trusted_environment_freeze.json',{k:'YES' if k=='A800_TRAINING_ENVIRONMENT_TRUSTED' else 'PASS' for k in ('A800_HOST_QUALIFICATION','FULL_MODEL_GPU_PARAMETER_INTEGRITY','FULL_MODEL_FRESH_PROCESS_REPEATABILITY','FP_RUNTIME_STABILITY','A800_TRAINING_ENVIRONMENT_TRUSTED')})
                write(root/'analysis/C5R_REPRODUCTION_GATE.json',{'C5R_REPRODUCTION':'INCONCLUSIVE','C5R_AIME26_ROUTE':'SKIP_INCONCLUSIVE'})
                for a in P.ALL_ARMS:
                    write(root/f'training/{a}/summary.json',{'status':'PASS','selected_update':63})
                    for kind in ('gradient_smoke','h64_readiness'):write(root/f'readiness/{a.lower()}_{kind}.json',{'status':'PASS'})
                comps=[]
                for a,b in P.MANDATORY_PAIRS:
                    comps.append({'comparison':a+'_vs_'+b,'net':net if b=='C6R' else hnet if a=='C7' and b=='H' else 0,'holm_p':1,'exact_two_sided_p':1,'both_correct':1,'both_wrong':1,'reference_only':1,'candidate_only':1,'cluster_bootstrap_ci95':[-.1,.1]})
                scored={a:{'correct':36,'accuracy':.6} for a in ('C6R','C7')}
                with patch.object(P,'ROOT',root),patch.object(P,'ARMS',('C6R','C7')):P.final(scored,comps)
                status=P.read(root/'analysis/final_status.json')
                self.assertEqual(status['PRIMARY_VERDICT'],label)
                self.assertEqual(status['C5R_AIME26'],'NOT_RUN')
                self.assertEqual(status['CLEANROOM_C6_MATERIAL_SHIFT'],'UNKNOWN')
                self.assertEqual(status['PRIMARY_MATCHED_COMPARISON'],'C7_VS_C6R')

if __name__=='__main__':unittest.main()
