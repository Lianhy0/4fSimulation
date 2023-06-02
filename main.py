import numpy as np
import matplotlib.pyplot as plt
from DiffLayer import DiffLayer
from Lens import Lens
# 注意这里衍射只算到复振幅，不算强度图

pixelNum = 1600
pixelSize = 7.6    # miu m
padingNum = 1600
waveLength = 0.640
lensFocal1 = 300000
lensRadius1 = 25400
lensFocal2 = 75000
lensRadius2 = 25400
tiltangle = np.pi * 0
k = 2 * np.pi / waveLength

diffDist0 = 300000
diffDist1 = 300000
diffDist2 = 75000
diffDist3 = 75000

# 初始化相位
phase, _ = np.meshgrid(np.arange(0, pixelNum, 1), np.arange(0, pixelNum, 1))
phase = phase * pixelSize * np.tan(tiltangle) * k

# 初始化复振幅
dmdComAmp = np.ones((pixelNum, pixelNum)) * np.exp(1.0j * phase)

# pad：
dmdComAmpPad = np.pad(dmdComAmp, padingNum, mode='constant')

# 计算复振幅衍射
diff1 = DiffLayer(int((pixelNum + 2 * padingNum)), pixelSize, diffDist0, waveLength)
eBeforeLens1 = diff1(dmdComAmpPad)

# 初始化透镜&经过透镜之后
lens1 = Lens(int((pixelNum + 2 * padingNum)), pixelSize, waveLength, lensFocal1, lensRadius1)
eAfterLens1 = lens1(eBeforeLens1)

# 衍射一段距离到焦点
diff2 = DiffLayer(int((pixelNum + 2 * padingNum)), pixelSize, diffDist1, waveLength)
eFocal1 = diff2(eAfterLens1)

# 从焦点到第二个透镜面
diff3 = DiffLayer(int((pixelNum + 2 * padingNum)), pixelSize, diffDist2, waveLength)
eBeforeLens2 = diff3(eFocal1)

# 第二个透镜&经过透镜之后
lens2 = Lens(int((pixelNum + 2 * padingNum)), pixelSize, waveLength, lensFocal2, lensRadius2)
eAfterLens2 = lens2(eBeforeLens2)

# 衍射一段距离
diff4 = DiffLayer(int((pixelNum + 2 * padingNum)), pixelSize, diffDist2, waveLength)
eFocal2 = diff4(eAfterLens2)

plt.figure(1)
plt.imshow((eFocal2[1600:3200, 1600:3200].real ** 2 + eFocal2[1600:3200, 1600:3200].imag ** 2))
plt.show()

# plt.figure(2)
# plt.imshow((eAfterLens2.real ** 2 + eAfterLens2.imag ** 2)[1600:3200, 1600:3200])
# plt.show()

print(0)