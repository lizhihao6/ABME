import os
import time

from imageio import imread, imwrite

from model.ABMENet import ABME

SAVE_DIR = "test_output"
if __name__ == "__main__":
    abme = ABME("cuda:1")
    start = time.clock()
    im0, im16 = imread("images/im1.png"), imread("images/im3.png")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    for i, im in enumerate(abme.xVFI(im0, im16, frame_num=16)):
        imwrite(os.path.join(SAVE_DIR, "{}.png".format(i)), im)
    end = time.clock()
    print("VFI time: {}".format(end - start))
