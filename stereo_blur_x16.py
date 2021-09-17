import os
import pickle
import sys
from multiprocessing.dummy import Pool
from shutil import copyfile

import numpy as np
from imageio import imread, imwrite
from tqdm import trange

from model.ABMENet import ABME

PATH = "/data/stereo_blur"
GPU_NUM = 8
BATCH_SIZE = 2


def get_ims():
    ims_path = "ims.pkl"
    if os.path.exists(ims_path):
        with open(ims_path, "rb+") as f:
            return pickle.load(f)
    dirs = [
        os.path.join(PATH, d, "image_left") for d in os.listdir(PATH)
        if os.path.isdir(os.path.join(PATH, d))
    ]
    dirs += [d.replace("left", "right") for d in dirs]
    ims = []
    for d in dirs:
        input_format = "%0{}d.png".format(len(os.listdir(d)[0]) - 4)
        im_ids = sorted(int(float(s[:-4])) for s in os.listdir(d))
        if not os.path.exists(d + "_x16"):
            os.makedirs(d + "_x16")
        for i in im_ids[:-1]:
            ims.append({
                "input": (os.path.join(d, input_format % i),
                          os.path.join(d, input_format % (i + 1))),
                "output": [
                    os.path.join(d + "_x16", "%05d.png" % _i)
                    for _i in range(i * 16, i * 16 + 16)
                ]
            })
    with open(ims_path, "wb+") as f:
        pickle.dump(ims, f)
    return ims


def x16(ims, mp_idx):
    step = int(np.ceil(float(len(ims)) / GPU_NUM))
    start_id, stop_id = step * mp_idx, min(step * mp_idx + step, len(ims))
    print(start_id, stop_id, mp_idx, flush=True)
    abme = ABME("cuda:{}".format(mp_idx), frame_num=16)
    iter = trange(start_id, stop_id, BATCH_SIZE) if start_id == 0 else range(
        start_id, stop_id, BATCH_SIZE)
    for i in iter:
        batch_size = BATCH_SIZE if i + BATCH_SIZE < stop_id else stop_id - 1 - i
        batch_im0 = [ims[i + j]["input"][0] for j in range(batch_size)]
        batch_im16 = [ims[i + j]["input"][1] for j in range(batch_size)]
        batch_outputs = [ims[i + j]["output"] for j in range(batch_size)]
        for b in range(batch_size):
            copyfile(batch_im0[b], batch_outputs[b][0])
        batch_im0, batch_im16 = [imread(p) for p in batch_im0
                                 ], [imread(p) for p in batch_im16]
        for b, _ims in enumerate(abme.xVFI(batch_im0, batch_im16)):
            for p, im in zip(batch_outputs[b][1:], _ims[1:]):
                imwrite(p, im)


def dist_x16(ims):
    print("gpu cores: {}".format(GPU_NUM))
    gpu_pool = Pool(GPU_NUM)
    results = [gpu_pool.apply_async(x16, args=(ims, i,)) for i in range(GPU_NUM)]
    results = [p.get() for p in results]


if __name__ == '__main__':
    idx = int(sys.argv[1])
    ims = get_ims()
    assert len(ims) % 4 == 0
    dist_x16(ims[idx * len(ims) // 4:(idx + 1) * len(ims) // 4])
