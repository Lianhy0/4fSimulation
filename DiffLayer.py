import numpy as np
import torch.nn as nn

class DiffLayer(nn.Module):
    def __init__(self, pixelNum, pixelSize, diffDist, waveLength):
        super(DiffLayer, self).__init__()
        self.size = pixelNum
        self.dL = self.size * pixelSize
        self.df = 1.0 / self.dL
        self.dist = diffDist
        self.lmb = waveLength
        self.k = np.pi * 2.0 / self.lmb
        self.H_z = self.InitH()

    def InitH(self):
        N = self.size
        df = self.df
        k = self.k
        d = self.dist
        lmb = self.lmb
        def phase(i, j):
            i -= N // 2
            j -= N // 2
            return ((i * df) * (i * df) + (j * df) * (j * df))

        ph = np.fromfunction(phase, shape=(N, N), dtype=np.float32)
        H = np.exp(1.0j * k * d) * np.exp(-1.0j * lmb * np.pi * d * ph)
        # H = np.exp(-1.0j * 2 * np.pi * d * np.sqrt((1 / lmb) * (1 / lmb) - ph))
        # H = np.exp(1.0j * k * d * (1 - lmb * lmb * 0.5 * ph))
        return H

    def forward(self, x):
        x11 = np.fft.fftshift(np.fft.fft2(x))
        x22 = x11 * self.H_z
        x33 = np.fft.ifft2(np.fft.ifftshift(x22))
        # xampp = x33.real * x33.real + x33.imag * x33.imag
        return x33