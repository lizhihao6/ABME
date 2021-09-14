import os
import time

from imageio import imread, imwrite

from model.ABMENet import ABME

SAVE_DIR = "test_output"
if __name__ == "__main__":
    abme = ABME("cuda:1", frame_num=16)
    start = time.clock()
    im0, im16 = imread("images/im1.png"), imread("images/im3.png")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    for i, im in enumerate(abme.xVFI([im0], [im16])):
        imwrite(os.path.join(SAVE_DIR, "{}.png".format(i)), im[0])
    end = time.clock()
    print("VFI time: {}".format(end - start))
