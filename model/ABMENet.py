from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn

from utils import warp
from .ABMNet import ABMRNet
from .SBMNet import SBMENet
from .SynthesisNet import SynthesisNet

cudnn.benchmark = True


class FAKE_ARGS:
    def __init__(self):
        self.DDP = False


class ABME(torch.nn.Module):
    def __init__(self, device, frame_num=16):
        super(ABME, self).__init__()

        self.frame_num = frame_num
        idx = [0, frame_num]
        run_idx = []
        while (idx[1] != 1):
            _idx = []
            for i in range(len(idx) - 1):
                input_idx0, input_idx1 = idx[i], idx[i + 1]
                tar_idx = (input_idx0 + input_idx1) // 2
                run_idx.append((input_idx0, input_idx1, tar_idx))
                _idx.append(tar_idx)
            idx = sorted(idx + _idx)
        self.run_idx = run_idx

        SBMNet = SBMENet()
        ABMNet = ABMRNet()
        args = FAKE_ARGS()
        SynNet = SynthesisNet(args)

        SBMNet.load_state_dict(
            torch.load('Best/SBME_ckpt.pth', map_location='cpu'))
        ABMNet.load_state_dict(
            torch.load('Best/ABMR_ckpt.pth', map_location='cpu'))
        SynNet.load_state_dict(
            torch.load('Best/SynNet_ckpt.pth', map_location='cpu'))

        for param in SBMNet.parameters():
            param.requires_grad = False
        for param in ABMNet.parameters():
            param.requires_grad = False
        for param in SynNet.parameters():
            param.requires_grad = False

        SBMNet.eval()
        ABMNet.eval()
        SynNet.eval()
        self.SBMNet, self.ABMNet, self.SynNet = SBMNet.to(device), ABMNet.to(
            device), SynNet.to(device)
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
            frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 *
                                                    (1 - Mask2_1) + frame2_1 *
                                                    (1 - Mask2_3))

            Z = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
            Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)

            ABM_bw, _ = self.ABMNet(torch.cat((frame2_Anchor, frame1_), dim=1),
                                    SBM * (-1), Z_.exp())
            ABM_fw, _ = self.ABMNet(torch.cat((frame2_Anchor, frame3_), dim=1),
                                    SBM, Z_.exp())

            SBM_ = F.interpolate(SBM, (H, W), mode='bilinear') * 20.0
            ABM_fw = F.interpolate(ABM_fw, (H, W), mode='bilinear') * 20.0
            ABM_bw = F.interpolate(ABM_bw, (H, W), mode='bilinear') * 20.0

            SBM_[:, 0, :, :] *= W / float(W_)
            SBM_[:, 1, :, :] *= H / float(H_)
            ABM_fw[:, 0, :, :] *= W / float(W_)
            ABM_fw[:, 1, :, :] *= H / float(H_)
            ABM_bw[:, 0, :, :] *= W / float(W_)
            ABM_bw[:, 1, :, :] *= H / float(H_)

            divisor = 8.
            H_ = int(ceil(H / divisor) * divisor)
            W_ = int(ceil(W / divisor) * divisor)

            Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw),
                                   dim=1)

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
    def _ims_to_tensor(ims):
        assert isinstance(ims, list)
        ims = np.concatenate([im[np.newaxis, ...] for im in ims], axis=0)
        batch_ims = torch.from_numpy(ims.astype(np.float32) / 255.).permute(
            [0, 3, 1, 2])
        return batch_ims

    @staticmethod
    def _tensor_to_ims(tensors):
        assert len(tensors.shape) == 5  # [frame_num+1, batch, c, h, w]
        batch_ims = (tensors * 255).detach().cpu().permute(0, 1, 3, 4,
                                                           2).numpy()
        batch_ims = np.clip(batch_ims, 0, 255.).astype(np.uint8)
        batch_ims = np.split(batch_ims, batch_ims.shape[1], axis=1)
        ims = [[im[0] for im in np.split(b[:, 0], b.shape[0], axis=0)] for b in batch_ims]
        return ims

    def xVFI(
            self,
            batch_im0,
            batch_imx,
    ):
        assert isinstance(batch_im0, list)
        with torch.no_grad():
            batch_frames = torch.zeros([
                self.frame_num + 1,
                len(batch_im0), 3, batch_im0[0].shape[0], batch_im0[0].shape[1]
            ]).to(self.device)
            batch_frames[0], batch_frames[-1] = ABME._ims_to_tensor(
                batch_im0).to(self.device), ABME._ims_to_tensor(batch_imx).to(
                self.device)
            for (input_idx0, input_idx1, tar_idx) in self.run_idx:
                batch_frames[tar_idx] = self.forward(batch_frames[input_idx0],
                                                     batch_frames[input_idx1])

            ims = ABME._tensor_to_ims(batch_frames[:-1])

        return ims
