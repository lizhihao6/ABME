import torch

from model import ABME

SAVE_PATH = "abme.jit"
if __name__ == '__main__':
    abme = ABME("cpu")
    frame1, frame3 = torch.randn([1, 3, 1280, 720]), torch.randn([1, 3, 1280, 720])
    trace_model = torch.jit.trace(abme, (frame1, frame3))
    trace_model.save(SAVE_PATH)
    traced_model = torch.jit.load(SAVE_PATH)
    print("filename : ", SAVE_PATH)
    print(traced_model.graph)
