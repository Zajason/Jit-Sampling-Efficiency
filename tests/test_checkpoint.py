from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from jit_pilot.upstream import ROOT, load_model

class TinyModel(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('buffer',torch.zeros(1))

class CheckpointTests(unittest.TestCase):
    def test_first_ema_overrides_parameters_and_base_buffers_survive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pth'
            torch.save({'model':{'weight':torch.tensor([1.]),'buffer':torch.tensor([7.])},
                        'model_ema1':{'weight':torch.tensor([3.])},'args':Namespace(model='JiT-B/16')},p)
            with patch('jit_pilot.upstream.denoiser_class',return_value=TinyModel):
                m=load_model(ROOT/'vendor/official-jit',p,torch.device('cpu'))
            self.assertEqual(m.weight.item(),3.)
            self.assertEqual(m.buffer.item(),7.)
            self.assertFalse(m.training)
            self.assertFalse(m.weight.requires_grad)

    def test_raw_weights_and_missing_ema_are_rejected(self):
        for checkpoint in [{'weight':torch.ones(1)}, {'model':{'weight':torch.ones(1),'buffer':torch.ones(1)},'model_ema1':{}}]:
            with self.subTest(checkpoint=list(checkpoint)):
                with tempfile.TemporaryDirectory() as d:
                    p=Path(d)/'checkpoint.pth';torch.save(checkpoint,p)
                    with patch('jit_pilot.upstream.denoiser_class',return_value=TinyModel):
                        with self.assertRaises(ValueError):load_model(ROOT/'vendor/official-jit',p,torch.device('cpu'))

if __name__=='__main__':unittest.main()
