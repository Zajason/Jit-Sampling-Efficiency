import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from jit_pilot.samplers import CONFIGS, Configuration, sample, ab2_update, expected_network_passes
from jit_pilot.upstream import ROOT, denoiser_class, verify_source
from jit_pilot.run import labels_for, noise_for, to_images

class SamplerTests(unittest.TestCase):
    def test_ab2_integrates_linear_time_velocity_on_variable_grid(self):
        # Integral of v(t)=2t+3 on [0,1] is 4. Supply exact history at t=.1.
        grid=[0.,.1,.23,.4,.68,1.]
        z=.31;previous=3.;previous_step=.1
        for i in range(1,len(grid)-1):
            t=grid[i];h=grid[i+1]-t;v=2*t+3
            z=ab2_update(z,v,previous,h,previous_step)
            previous,previous_step=v,h
        self.assertAlmostEqual(z,4.,places=12)

    def test_ab2_second_order_convergence(self):
        def integrate(n):
            h=1/n;z=1.;previous=None
            for i in range(n):
                velocity=z;z=ab2_update(z,velocity,previous,h,h);previous=velocity
            return abs(z-math.e)
        ratio=integrate(40)/integrate(80)
        self.assertGreater(ratio,3.7)
        self.assertLess(ratio,4.2)

    def test_pairing_independent_of_batch_boundaries(self):
        all_noise=noise_for(range(5),42,size=4)
        split=torch.cat([noise_for(range(2),42,size=4),noise_for(range(2,5),42,size=4)])
        self.assertTrue(torch.equal(all_noise,split))
        labels=labels_for(2000,42)
        self.assertEqual(len(set(labels[:1000])),1000)
        self.assertEqual(labels[:1000],labels[1000:])

    def test_output_rounding_and_nonfinite_rejection(self):
        z=torch.tensor([[[[-1.,0.,1.,2.]]]]).expand(1,3,1,4)
        self.assertEqual(to_images(z)[0,0,:,0].tolist(),[0,128,255,255])
        with self.assertRaises(RuntimeError):to_images(torch.full((1,3,1,1),float('nan')))

class OfficialParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'vendor/official-jit'
        verify_source(path)
        cls.Denoiser=denoiser_class(path)

    def fake_model(self):
        # Actual official Denoiser sampler/guidance code with a tiny analytic network.
        # No pretrained-image claim is made by these numerical tests.
        class Network(torch.nn.Module):
            def forward(self,z,t,labels):
                return .2*z + .1*t.view(-1,1,1,1) + .0001*labels.view(-1,1,1,1)
        model=self.Denoiser.__new__(self.Denoiser)
        torch.nn.Module.__init__(model)
        model.net=Network();model.img_size=4;model.noise_scale=1.;model.num_classes=1000
        model.cfg_scale=3.;model.cfg_interval=(.1,1.);model.t_eps=.05
        return model

    def test_euler_heun_match_official_generate(self):
        for method in ['euler','heun']:
            for steps in [1,2,25,50]:
                with self.subTest(method=method,steps=steps):
                    model=self.fake_model();model.method=method;model.steps=steps
                    labels=torch.tensor([1,702])
                    torch.manual_seed(10);expected=model.generate(labels)
                    torch.manual_seed(10);z=torch.randn(2,3,4,4)
                    actual=sample(model,z,labels,Configuration('test',method,steps))
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_instrumented_call_counts_and_noise_preservation(self):
        model=self.fake_model();labels=torch.tensor([3,93]);z=torch.randn(2,3,4,4);copy=z.clone()
        for config in CONFIGS:
            calls=[]
            handle=model.net.register_forward_pre_hook(lambda *args:calls.append(1))
            try:result=sample(model,z,labels,config)
            finally:handle.remove()
            self.assertEqual(len(calls),expected_network_passes(config.method,config.steps))
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(z,copy,rtol=0,atol=0)

    def test_nonuniform_grids_integrate_linear_time_velocity(self):
        model=self.fake_model()
        model._forward_sample=lambda z,t,y: (2*t+3).expand_as(z)
        labels=torch.tensor([3,93]);z=torch.zeros(2,3,4,4)
        for power in [.5,2.]:
            grid=torch.linspace(0.,1.,26).pow(power)
            for method in ['euler','heun','ab2']:
                actual=sample(model,z,labels,Configuration('test',method,25,power))
                h=grid[1:]-grid[:-1]
                if method=='euler': expected=float(((2*grid[:-1]+3)*h).sum())
                elif method=='heun': expected=4-float(h[-1]**2)
                else: expected=4-float(h[0]**2)
                torch.testing.assert_close(actual,torch.full_like(z,expected),rtol=1e-5,atol=1e-5)

    def test_invalid_schedules_are_rejected(self):
        for power in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):Configuration('bad','ab2',25,power)

if __name__=='__main__':unittest.main()
