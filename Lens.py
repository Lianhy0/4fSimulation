import numpy as np
import torch.nn as nn

class Lens(nn.Module):
    def __init__(self, pixelNum, pixelSize, waveLength, focal, radius):
        super(Lens, self).__init__()
        self.pixelNum = pixelNum
        self.pixelSize = pixelSize
        self.size = pixelNum
        self.dL = self.size * pixelSize
        self.lmb = waveLength
        self.focal = focal
        self.radius = radius
        self.k = np.pi * 2 / self.lmb
        self.H = self.InitH()

    def InitH(self):
        hh = np.arange(-(self.pixelNum // 2), self.pixelNum // 2, 1)
        ww = np.arange(-(self.pixelNum // 2), self.pixelNum // 2, 1)
        arrH, arrW = np.meshgrid(hh, ww)
        arrCircle = ((arrH * self.pixelSize) ** 2 + (arrW * self.pixelSize) ** 2)
        arrCircle[arrCircle > (self.radius ** 2)] = 0
        transFunc = np.exp(-1.0j * self.k * arrCircle / 2 / self.focal)

        return transFunc

    def forward(self, e):
        eAfterLens = self.H * e
        return eAfterLens


if __name__ == '__main__':
    lens = Lens(4800, 7.6, 0.640, 300000, 25400)