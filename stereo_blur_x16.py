import multiprocessing as mp
import os
from shutil import copyfile

import numpy as np
from imageio import imread, imwrite
from tqdm import trange

from model.ABMENet import ABME

PATH = "/data/stereo_blur"
GPU_NUM = 8
BATCH_SIZE = 2


def son_imread(idx, path):
    return idx, imread(path)


def dist_imread(pool, paths):
    results = [
        pool.apply_async(son_imread, args=(i, p)) for i, p in enumerate(paths)
    ]
    results = [p.get() for p in results]
    return [r[1] for r in sorted(results, key=lambda x: x[0])]


def dist_imwrite(pool, paths, ims):
    results = [
        pool.apply_async(imwrite, args=(p, im)) for p, im in zip(paths, ims)
    ]
    results = [p.get() for p in results]


def x16(ims, pool):
    mp_idx = os.getpid() % GPU_NUM
    step = int(np.ceil(float(len(ims)) / GPU_NUM))
    start_id, stop_id = step * mp_idx, min(step * mp_idx + step, len(ims))
    abme = ABME("cuda:{}".format(mp_idx), frame_num=16)
    iter = trange(start_id, stop_id, BATCH_SIZE) if start_id == 0 else range(
        start_id, stop_id, BATCH_SIZE)
    for i in iter:
        if start_id == 0:
            print("{}/{}".format(i, stop_id), flush=True)
        batch_im0 = [ims[i + j]["input"][0] for j in range(BATCH_SIZE)]
        batch_im16 = [ims[i + j]["input"][1] for j in range(BATCH_SIZE)]
        batch_outputs = [ims[i + j]["output"] for j in range(BATCH_SIZE)]
        for b in range(BATCH_SIZE):
            copyfile(batch_im0[b], batch_outputs[b][0])
        # batch_im0 = dist_imread(pool, batch_im0)
        # batch_im16 = dist_imread(pool, batch_im16)
        batch_im0, batch_im16 = [imread(p) for p in batch_im0
                                 ], [imread(p) for p in batch_im16]
        for b, ims in enumerate(abme.xVFI(batch_im0, batch_im16)):
            dist_imwrite(pool, batch_outputs[b][1:], ims[1:])
    for i in range((stop_id - start_id) // BATCH_SIZE * BATCH_SIZE + start_id,
                   stop_id):
        batch_im0, batch_im16 = [imread(ims[i]["input"][0])
                                 ], [imread(ims[i]["input"][1])]
        ims = abme.xVFI(batch_im0, batch_im16)[0]
        dist_imwrite(pool, [ims[i]["output"]], ims)


def dist_x16(ims):
    cpu_num = int(mp.cpu_count()) // GPU_NUM
    assert cpu_num > 0
    print("cpu cores: {}".format(cpu_num))
    cpu_pool = mp.Pool(cpu_num)

    print("gpu cores: {}".format(GPU_NUM))
    gpu_pool = mp.Pool(GPU_NUM)
    results = [gpu_pool.apply_async(x16, args=(ims, cpu_pool))]
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
