"""CPU-only numerical contracts; synthetic non-AIME fixtures only."""
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('wscale_math_reference',ROOT/'reference/math_reference.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


def state(fn=lambda h,k,v:0.0, heads=1):
    return M.Tensor32((1,heads,128,128),
        [fn(h,k,v) for h in range(heads) for k in range(128) for v in range(128)])


def weights(fn=lambda h,k:1.0,heads=1):
    return M.Tensor32((heads,128),[fn(h,k) for h in range(heads) for k in range(128)])


class MathReferenceTests(unittest.TestCase):
    def test_01_original_peak_floor_not_scale_floor(self):
        for graph in ('CPU_CANONICAL','CUDA_LEGACY_EMULATED'):
            s=M.canonical_scale(0,graph)
            self.assertGreater(s,0)
            self.assertAlmostEqual(s,1e-12/127,delta=1e-20)
            self.assertLess(s,1e-13)

    def test_02_original_division_graph_is_explicit(self):
        # Proven counterexample: CPU division and CUDA reciprocal multiply
        # need not have the same last bit, despite the same real formula.
        p=M.f32(0.1134604886174202)
        a=M.canonical_scale(p,'CPU_CANONICAL')
        b=M.canonical_scale(p,'CUDA_LEGACY_EMULATED')
        self.assertEqual(a,M.div(p,127))
        self.assertEqual(b,M.mul(p,M.f32(1/127)))
        with self.assertRaises(ValueError):M.canonical_scale(1,'UNSPECIFIED')

    def test_03_nearest_even(self):
        self.assertEqual([M.sym8_scalar(v,1)[0] for v in (.5,1.5,2.5,-.5,-1.5,-2.5)],
                         [0,2,2,0,-2,-2])

    def test_04_signed_zero(self):
        for value in (-0.0,-0.1):
            code,flag,d=M.sym8_scalar(value,1)
            self.assertEqual(code,0);self.assertTrue(flag)
            self.assertEqual(math.copysign(1,d),-1)
        self.assertFalse(M.sym8_scalar(0.0,1)[1])

    def test_05_signed_int8_range_and_clipping(self):
        self.assertEqual(M.sym8_scalar(10000,1)[0],127)
        self.assertEqual(M.sym8_scalar(-10000,1)[0],-127)
        self.assertNotEqual(M.sym8_scalar(-10000,1)[0],-128)

    def test_06_floor_for_all_candidates(self):
        for c in M.LAMBDAS:
            self.assertGreaterEqual(M.candidate_scale(M.canonical_scale(0),c),M.canonical_scale(0))

    def test_07_exact_tie_order(self):
        self.assertEqual(M.choose_candidate([1,1,1,1,1]),3)
        self.assertEqual(M.choose_candidate([1,1,0,1,0]),2)
        self.assertEqual(M.choose_candidate([0,0,1,1,1]),1)

    def test_08_zero_state_search_selects_lambda_one(self):
        e=M.quantize(state(),weights(),'wscale')
        self.assertEqual(set(e.candidate_indices),{3})
        self.assertEqual(set(e.codes),{0})
        self.assertEqual(set(e.losses.values),{0})

    def test_09_positive_and_negative_constant(self):
        for sign in (-1,1):
            e=M.quantize(state(lambda h,k,v:sign*.5),weights(),'base')
            self.assertEqual(set(e.codes),{sign*127})
            self.assertTrue(all(math.isfinite(x) for x in M.decode(e).values))

    def test_10_small_dynamic_range(self):
        e=M.quantize(state(lambda h,k,v:(k-64)*1e-17),weights(),'wscale')
        self.assertTrue(all(s>0 for s in e.scales.values))
        self.assertTrue(all(math.isfinite(x) for x in M.decode(e).values))

    def test_11_group_key_axis_not_value_axis(self):
        x=state(lambda h,k,v:float(k//32+1)*(v+1))
        e=M.quantize(x,weights(),'base')
        for g in range(4):
            for v in (0,31,127):
                self.assertEqual(e.scales.values[g*128+v],M.canonical_scale((g+1)*(v+1)))

    def test_12_head_column_isolation(self):
        x=state(lambda h,k,v:(h+1)*100+(k//32)*10+v*.125,heads=2)
        e=M.quantize(x,weights(heads=2),'base')
        self.assertNotEqual(e.scales.values[0],e.scales.values[512])
        self.assertNotEqual(e.scales.values[0],e.scales.values[127])

    def test_13_lambda_one_r_one_exact_base(self):
        x=state(lambda h,k,v:math.sin((k*128+v)*.123)*1.7)
        w=weights(lambda h,k:2.0**((k%9)-4))
        for graph in ('CPU_CANONICAL','CUDA_LEGACY_EMULATED'):
            base=M.quantize(x,w,'base',arithmetic=graph)
            for method in ('wscale','rowbal','rowbal_wscale'):
                e=M.quantize(x,w,method,arithmetic=graph,force_unit_row=True,force_lambda_one=True)
                self.assertEqual(e.codes.tobytes(),base.codes.tobytes())
                self.assertEqual(e.scales.bytes(),base.scales.bytes())
                self.assertEqual(e.negative_zero,base.negative_zero)
                self.assertEqual(M.decode(e).bytes(),M.decode(base).bytes())

    def test_14_row_factor_uniformity(self):
        r=M.row_factors(state(lambda h,k,v:.5),weights(),.5,.25)
        self.assertTrue(all(abs(x-1)<2e-7 for x in r.values))

    def test_15_row_factors_positive_finite_clamped(self):
        x=state(lambda h,k,v:0 if k%4==0 else 10**((k%7)-3))
        w=weights(lambda h,k:16 if k%2 else 1/16)
        r=M.row_factors(x,w,.5,.5)
        self.assertTrue(all(.25<=v<=4 for v in r.values))
        self.assertIn(.25,r.values);self.assertIn(4,r.values)

    def test_16_rowbal_and_combined_share_identical_r(self):
        x=state(lambda h,k,v:math.sin(k*.7+v*.12)*(k+1)/32)
        w=weights(lambda h,k:1/16+(k%16))
        r=M.quantize(x,w,'rowbal',a=.5,b=.25)
        rw=M.quantize(x,w,'rowbal_wscale',a=.5,b=.25)
        self.assertEqual(r.row_factors.bytes(),rw.row_factors.bytes())
        self.assertTrue(all(a<=b for a,b in zip(rw.losses.values,r.losses.values)))

    def test_17_wscale_loss_not_above_base(self):
        rng=random.Random(71)
        x=state(lambda h,k,v:rng.gauss(0,1)*(10 if k%31==0 else 1))
        w=weights(lambda h,k:1/16 if k%31==0 else 16)
        b=M.quantize(x,w,'base');e=M.quantize(x,w,'wscale')
        self.assertTrue(all(a<=bb for a,bb in zip(e.losses.values,b.losses.values)))
        self.assertTrue(any(i!=3 for i in e.candidate_indices))

    def test_18_combined_original_x_space_objective(self):
        # r^2 makes original-space loss different from plain normalized-Y MSE.
        # This fixture must distinguish the two candidate argmins.
        rng=random.Random(9321)
        found=False
        for _ in range(100):
            x=[M.f32(rng.gauss(0,1)*10**rng.uniform(-1,1)) for _ in range(32)]
            r=[M.f32(rng.uniform(.25,4)) for _ in x]
            w=[M.f32(2**rng.uniform(-4,4)) for _ in x]
            y=[M.div(a,b) for a,b in zip(x,r)]
            s0=M.canonical_scale(max(map(abs,y)))
            lx=[];ly=[]
            for candidate in M.LAMBDAS:
                s=M.candidate_scale(s0,candidate)
                yd=[M.sym8_scalar(v,s)[2] for v in y]
                dx=[M.add(v,-M.mul(d,rf)) for v,d,rf in zip(x,yd,r)]
                dy=[M.add(v,-d) for v,d in zip(y,yd)]
                lx.append(M.tree_sum([M.mul(ww,M.mul(d,d)) for ww,d in zip(w,dx)]))
                ly.append(M.tree_sum([M.mul(ww,M.mul(d,d)) for ww,d in zip(w,dy)]))
            if M.choose_candidate(lx)!=M.choose_candidate(ly):found=True;break
        self.assertTrue(found,'fixture must catch normalized-space loss bug')

    def test_19_reject_nonfinite_and_bad_shape(self):
        for bad in (math.nan,math.inf,-math.inf):
            x=state();x.values[0]=bad
            with self.assertRaises(FloatingPointError):M.quantize(x,weights(),'base')
        with self.assertRaises(ValueError):M.quantize(M.Tensor32((128,128),[0]*16384),weights(),'base')
        with self.assertRaises(ValueError):M.quantize(state(),M.Tensor32((2,128),[1]*256),'base')

    def test_20_reject_bad_omega(self):
        for bad in (0,-1):
            w=weights();w.values[0]=bad
            with self.assertRaises(ValueError):M.quantize(state(),w,'wscale')
        with self.assertRaises(FloatingPointError):M.choose_candidate([0,1,2,3,math.nan])

    def test_21_calibration_weight_stabilization(self):
        raw=weights(lambda h,k:0 if k%32==0 else 10**((k%15)-7),heads=2)
        w,audit=M.stabilize_omega(raw)
        self.assertTrue(all(1/16<=v<=16 for v in w.values))
        self.assertGreater(audit['clamp_low_count'],0)
        self.assertGreater(audit['clamp_high_count'],0)
        zero,_=M.stabilize_omega(weights(lambda h,k:0))
        self.assertTrue(all(abs(v-1)<2e-6 for v in zero.values))

    def test_22_no_head_copy_from_shared_query(self):
        q=[.02]*128;k=[.05]*128
        a=M.transported_query(q,k,-.1,.2)
        b=M.transported_query(q,k,-1.1,.8)
        self.assertNotEqual(a,b)

    def test_23_transported_query_matches_dense_transition(self):
        q=[0.0]*128;k=[0.0]*128;q[0]=2;q[1]=3;k[0]=.6;k[1]=.8
        beta=.3;log_gate=-.2
        g=M.transported_query(q,k,log_gate,beta)
        alpha=math.exp(log_gate);scale=1/math.sqrt(128)
        for i in range(128):
            expected=alpha*(q[i]*scale-beta*k[i]*sum(ki*qi*scale for ki,qi in zip(k,q)))
            self.assertAlmostEqual(g[i],expected,delta=2e-8)

    def test_24_actual_update_error_propagates_via_transport(self):
        # E.T @ A.T @ q is the exact real-arithmetic change in next readout.
        rng=random.Random(11)
        q=[rng.uniform(-.1,.1) for _ in range(128)]
        k=[rng.uniform(-.05,.05) for _ in range(128)]
        e=[rng.uniform(-.01,.01) for _ in range(128)]
        beta=.7;alpha=.91
        g=M.transported_query(q,k,math.log(alpha),beta,query_already_scaled=True)
        propagated=[alpha*(ei-beta*ki*sum(kk*ee for kk,ee in zip(k,e))) for ki,ei in zip(k,e)]
        lhs=sum(qi*pi for qi,pi in zip(q,propagated));rhs=sum(ei*gi for ei,gi in zip(e,g))
        self.assertAlmostEqual(lhs,rhs,delta=2e-9)

    def test_25_immediate_writeback_and_signed_zero(self):
        x=state(lambda h,k,v:-0.0 if k==0 else -.001 if k==1 else .5)
        e=M.qdq_writeback(x,weights(),'rowbal_wscale',a=.5,b=.25)
        self.assertEqual(x.bytes(),M.decode(e).bytes())
        self.assertEqual(math.copysign(1,x.values[0]),-1)

    def test_26_frozen_parameter_grid_only(self):
        with self.assertRaises(ValueError):M.quantize(state(),weights(),'rowbal',a=.2,b=.3)
        with self.assertRaises(ValueError):M.quantize(state(),weights(),'rowbal')
        with self.assertRaises(ValueError):M.transported_query([0]*128,[0]*128,.2,.5)

    def test_27_independent_fields_given_validated_row_inputs(self):
        x=state(lambda h,k,v:math.sin(k*.5+v*.03)*(1+k%9))
        w=weights(lambda h,k:2.**((k%7)-3))
        r=M.row_factors(x,w,.5,.25)
        ordinary=M.quantize(x,w,'rowbal_wscale',a=.5,b=.25)
        separated=M.quantize(x,w,'rowbal_wscale',a=.5,b=.25,row_factors_override=r)
        self.assertEqual(ordinary.codes.tobytes(),separated.codes.tobytes())
        self.assertEqual(ordinary.scales.bytes(),separated.scales.bytes())
        self.assertEqual(ordinary.candidate_indices,separated.candidate_indices)
        self.assertEqual(M.decode(ordinary).bytes(),M.decode(separated).bytes())
        with self.assertRaises(ValueError):M.quantize(x,w,'base',row_factors_override=r)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MathReferenceTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if '--json' in sys.argv:
        p=Path(sys.argv[sys.argv.index('--json')+1])
        p.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            scope='CPU scalar FP32 reference, synthetic non-AIME mathematical fixtures; no GPU proof'),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
