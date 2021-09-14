import multiprocessing as mp
import os
from shutil import copyfile

import numpy as np
from imageio import imread, imwrite
from tqdm import trange

from model.ABMENet import ABME

PATH = "/data/stereo_blur"
GPU_NUM = 8


def x16(ims):
    mp_idx = os.getpid() % GPU_NUM
    step = int(np.ceil(float(len(ims)) / GPU_NUM))
    start_id, stop_id = step * mp_idx, min(step * mp_idx + step, len(ims))
    abme = ABME("cuda:{}".format(mp_idx))
    iter = trange(start_id, stop_id) if start_id == 0 else range(
        start_id, stop_id)
    for i in iter:
        if start_id == 0:
            print("{}/{}".format(i, stop_id), flush=True)
        input, output = ims[i]["input"], ims[i]["output"]
        copyfile(input[0], output[0])
        for p, im in zip(
                output[1:],
                abme.xVFI(imread(input[0]), imread(input[1]),
                          frame_num=16)[1:]):
            imwrite(p, im)


def dist_x16(ims):
    num_cores = GPU_NUM
    print("num cores: {}".format(num_cores))
    pool = mp.Pool(num_cores)
    results = [pool.apply_async(x16, args=(ims, ))]
    results = [p.get() for p in results]


if __name__ == '__main__':
    dirs = [
        os.path.join(PATH, d, "image_left") for d in os.listdir(PATH)
        if os.path.isdir(os.path.join(PATH, d))
    ]
    dirs += [d.replace("left", "right") for d in dirs]
    ims = []
    for d in dirs:
        im_ids = sorted(int(float(s[:-4])) for s in os.listdir(d))
        if not os.path.exists(d + "_x16"):
            os.makedirs(d + "_x16")
        for i in im_ids[:-1]:
            ims.append({
                "input": (os.path.join(d, "%04d.png" % i),
                          os.path.join(d, "%04d.png" % (i + 1))),
                "output": [
                    os.path.join(d + "_x16", "%05d.png" % _i)
                    for _i in range(i * 16, i * 16 + 16)
                ]
            })
    dist_x16(ims)
