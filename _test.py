import multiprocessing as mp
import os
import time
from shutil import copyfile

from imageio import imread, imwrite
from cv2 import resize
from model.ABMENet import ABME

SAVE_DIR = "test_output"
GPU_NUM = 2


def dist_imwrite(pool, paths, ims):
    results = [
        pool.apply_async(imwrite, args=(p, im)) for p, im in zip(paths, ims)
    ]
    results = [p.get() for p in results]


if __name__ == "__main__":
    abme = ABME("cuda:1", frame_num=16)
    im0, im16 = imread("images/im1.png"), imread("images/im3.png")
    im0, im16 = resize(im0, (1280, 720)), resize(im16, (1280, 720))

    num_cores = int(mp.cpu_count()) // GPU_NUM
    assert num_cores > 0
    print("num cores: {}".format(num_cores))
    pool = mp.Pool(num_cores)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    output_paths = [os.path.join(SAVE_DIR, "{}.png".format(i)) for i in range(16)]
    start = time.clock()
    ims = abme.xVFI([im0], [im16])[0]
    copyfile("images/im1.png", output_paths[0])
    dist_imwrite(pool, output_paths[1:], ims[1:])
    end = time.clock()
    print("VFI time: {}".format(end - start))
