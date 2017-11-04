import itertools
from matplotlib import pyplot as plt
import cv2
import sys
import numpy as np
import peakutils

def calcSma(data, smaPeriod):
    j = next(i for i, x in enumerate(data) if x is not None)
    our_range = range(len(data))[j + smaPeriod - 1:]
    empty_list = [0.0] * (j + smaPeriod - 1)
    sub_result = [np.mean(data[i - smaPeriod + 1: i + 1]) for i in our_range]

    return np.array(empty_list + sub_result)

# Abre a imagem escolhida
image = cv2.imread('cores.jpeg')

# Converte a imagem para nivel de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist_img = cv2.calcHist([gray_image],[0],None,[256],[0,256])
smoothed_hist = calcSma(hist_img.ravel(), 5);
print smoothed_hist
smoothed_hist = hist_img.ravel()

indexes = peakutils.indexes(smoothed_hist, thres=0.12, min_dist=5)
#print indexes
plt.plot(range(256),smoothed_hist)
y_ind = np.zeros(256)
for i in indexes:
    y_ind[i] = smoothed_hist[i]
plt.plot(range(256), y_ind)

x_th = []
for i in xrange(len(indexes)-1):
    x_th.append((indexes[i]+indexes[i+1])/2)
y_th = np.zeros(256)
for x in x_th:
    y_th[x] = smoothed_hist[x]
plt.plot(range(256), y_th, marker='x')

print x_th

plt.show()
