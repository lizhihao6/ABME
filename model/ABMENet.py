from math import ceil
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from .SBMNet import SBMENet
from .ABMNet import ABMRNet
from .SynthesisNet import SynthesisNet
from utils import warp

cudnn.benchmark = True
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.DDP = False


class ABME(torch.nn.Module):

    def __init__(self, device):
        super(ABME, self).__init__()
        SBMNet = SBMENet()
        ABMNet = ABMRNet()
        SynNet = SynthesisNet(args)

        SBMNet.load_state_dict(torch.load('Best/SBME_ckpt.pth', map_location='cpu'))
        ABMNet.load_state_dict(torch.load('Best/ABMR_ckpt.pth', map_location='cpu'))
        SynNet.load_state_dict(torch.load('Best/SynNet_ckpt.pth', map_location='cpu'))

        for param in SBMNet.parameters():
            param.requires_grad = False
        for param in ABMNet.parameters():
            param.requires_grad = False
        for param in SynNet.parameters():
            param.requires_grad = False

        SBMNet.eval()
        ABMNet.eval()
        SynNet.eval()
        self.SBMNet, self.ABMNet, self.SynNet = SBMNet.to(device), ABMNet.to(device), SynNet.to(device)
        self.device = device

    def forward(self, frame1, frame3):
        with torch.no_grad():
            frame1, frame3 = frame1.to(self.device), frame3.to(self.device)
            H = frame1.shape[2]
            W = frame1.shape[3]

            divisor = 64.
            D_factor = 1.

            H_ = int(ceil(H / divisor) * divisor * D_factor)
            W_ = int(ceil(W / divisor) * divisor * D_factor)

            frame1_ = F.interpolate(frame1, (H_, W_), mode='bicubic')
            frame3_ = F.interpolate(frame3, (H_, W_), mode='bicubic')

            SBM = self.SBMNet(torch.cat((frame1_, frame3_), dim=1))[0]
            SBM_ = F.interpolate(SBM, scale_factor=4, mode='bilinear') * 20.0

            frame2_1, Mask2_1 = warp(frame1_, SBM_ * (-1), return_mask=True)
            frame2_3, Mask2_3 = warp(frame3_, SBM_, return_mask=True)

            frame2_Anchor_ = (frame2_1 + frame2_3) / 2
            frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 * (1 - Mask2_1) + frame2_1 * (1 - Mask2_3))

            Z = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
            Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)

            ABM_bw, _ = self.ABMNet(torch.cat((frame2_Anchor, frame1_), dim=1), SBM * (-1), Z_.exp())
            ABM_fw, _ = self.ABMNet(torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp())

            SBM_ = F.interpolate(SBM, (H, W), mode='bilinear') * 20.0
            ABM_fw = F.interpolate(self.ABM_fw, (H, W), mode='bilinear') * 20.0
            ABM_bw = F.interpolate(self.ABM_bw, (H, W), mode='bilinear') * 20.0

            SBM_[:, 0, :, :] *= W / float(W_)
            SBM_[:, 1, :, :] *= H / float(H_)
            ABM_fw[:, 0, :, :] *= W / float(W_)
            ABM_fw[:, 1, :, :] *= H / float(H_)
            ABM_bw[:, 0, :, :] *= W / float(W_)
            ABM_bw[:, 1, :, :] *= H / float(H_)

            divisor = 8.
            H_ = int(ceil(H / divisor) * divisor)
            W_ = int(ceil(W / divisor) * divisor)

            Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)

            Syn_inputs = F.interpolate(Syn_inputs, (H_, W_), mode='bilinear')
            Syn_inputs[:, 6, :, :] *= float(W_) / W
            Syn_inputs[:, 7, :, :] *= float(H_) / H
            Syn_inputs[:, 8, :, :] *= float(W_) / W
            Syn_inputs[:, 9, :, :] *= float(H_) / H
            Syn_inputs[:, 10, :, :] *= float(W_) / W
            Syn_inputs[:, 11, :, :] *= float(H_) / H

            result = self.SynNet(Syn_inputs)
            result = F.interpolate(result, (H, W), mode='bicubic')
            return result
    
    @staticmethod
    def _im_to_tensor(im):
        tensor = torch.from_numpy(im.astype(np.float32)/255.).permute([2, 0, 1]).unsqueeze(0)

    @staticmethod
    def _tensor_to_im(tensor):
        im = tensor.detach().cpu()[0].permute([1, 2, 0]).numpy()
        return np.clip(im, 0, 255.).astype(np.uint8)

    def x8(self, im0, im8):
        frame0, frame1 = ABME._im_to_tensor(im0), ABME._im_to_tensor(im8)
        with torch.no_grad():
            frame0, frame8 = frame0.to(self.device), frame8.to(self.device)
            frame4 = self.forward(frame0, frame8)
            frame2 = self.forward(frame0, frame4)
            frame6 = self.forward(frame4, frame8)
            frame1 = self.forward(frame0, frame2)
            frame3 = self.forward(frame2, frame4)
            frame5 = self.forward(frame4, frame6)
            frame7 = self.forward(frame6, frame8)
        ims = [im0] + [ABME._tensor_to_im(f) for f in [frame1, frame2, frame3, frame4, frame5, frame6, frame7]] + [im8]
        del frame0
        del frame1
        del frame2
        del frame3
        del frame4
        del frame5
        del frame6
        del frame7
        del frame8
        torch.cuda.empty_cache()
        return ims