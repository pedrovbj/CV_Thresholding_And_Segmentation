import cv2
from matplotlib import pyplot as plt
import numpy as np
import itertools

PATH = ['moedas.jpg', 'cores.jpeg']

def otsu_multi(img, numcl, maxval):
    hist = np.bincount(img.ravel(),minlength=maxval)
    bins = np.arange(maxval)
    Q = hist.cumsum()
    N = Q[-1]
    Q = np.float64(Q)/N
    hist = np.float64(hist)/N
    fn_max = -np.inf
    thresh = None
    for t in itertools.combinations(xrange(maxval), numcl-1):
        p = np.hsplit(hist, t)
        q = np.zeros(numcl)
        q[0] = Q[t[0]]
        for i in xrange(1, numcl-1):
            q[i] = Q[t[i]]-Q[t[i-1]]
        q[-1] = Q[-1]-Q[t[-1]]
        b = np.hsplit(bins, t)
        m = np.zeros(numcl)
        for i in xrange(numcl):
            m[i] = np.sum(p[i]*b[i])/q[i] if q[i] > 0.0 else 0.0
        fn = np.sum(q*m*m)
        if fn > fn_max:
            fn_max = fn
            thresh = t
    return thresh

def otsu_fast(img, numcl, maxval):
    hist = np.bincount(img.ravel(),minlength=maxval)
    hist = np.float64(hist)/np.sum(hist)
    P = np.zeros((maxval,maxval))
    S = np.zeros((maxval,maxval))
    H = np.zeros((maxval,maxval))
    P[0][0] = hist[0]
    S[0][0] = hist[0]
    for k in xrange(1, maxval):
            P[0][k] = P[0][k-1] + hist[k]
            S[0][k] = S[0][k-1] +(k+1)*hist[k]
    for i in xrange(1, maxval):
        for j in xrange(i, maxval):
            P[i][j] = P[0][j]-P[0][i-1]
            S[i][j] = S[0][j]-S[0][i-1]
    for i in xrange(0, maxval):
        for j in xrange(0, maxval):
            H[i][j] = (S[i][j]*S[i][j])/P[i][j] if P[i][j] > 0 else 0.0
    fn_max = -np.inf
    thresh = None
    for t in itertools.combinations(xrange(0,maxval-1), numcl-1):
        fn = H[0][t[0]]
        for k in xrange(1,numcl-1):
            fn += H[t[k-1]+1][t[k]]
        fn += H[t[-1]+1][-1]
        if fn > fn_max:
            fn_max = fn
            thresh = t
    return thresh

def apply_thresholds(img, thresh, value, maxval):
    thresh.insert(0, 0)
    thresh.append(maxval)
    h, w = img.shape
    retimg = np.ndarray((h,w), np.uint8)
    for y in xrange(h):
        for x in xrange(w):
            for i in xrange(len(thresh)-1):
                if img[y,x] > thresh[i] and img[y,x] <= thresh[i+1]:
                    retimg[y,x] = value[i-1]
    return retimg

def plot_gray(img1, img2, title):
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    plt.imshow(img1, cmap='gray')
    a=fig.add_subplot(1,2,2)
    plt.imshow(img2, cmap='gray')
    plt.title(title)
    plt.show()

def main():
    # Abre as imagens
    moedas_img = cv2.imread(PATH[0])
    cores_img = cv2.imread(PATH[1])

    # Converte as imagens para niveis de cinza
    moedas_gray_img = cv2.cvtColor(moedas_img, cv2.COLOR_BGR2GRAY)
    cores_gray_img = cv2.cvtColor(cores_img, cv2.COLOR_BGR2GRAY)

    # Aplica filtro Gaussiano
    moedas_gray_img = cv2.GaussianBlur(moedas_gray_img, (15,15), 0)
    cores_gray_img = cv2.GaussianBlur(cores_gray_img, (15,15), 0)

    # Limiariza a imagem a partir de um limiar manual
    retval, moedas_th_manual_gray_img = cv2.threshold(moedas_gray_img, 116, 255, cv2.THRESH_BINARY_INV)
    retval, cores_th_manual_gray_img = cv2.threshold(cores_gray_img, 140, 255, cv2.THRESH_BINARY_INV)
    # plot_gray(moedas_gray_img, moedas_th_manual_gray_img, 'Limiarizacao manual com 1 nivel')
    # plot_gray(cores_gray_img, cores_th_manual_gray_img, 'Limiarizacao manual com 1 nivel')

    # Limiariza a imagem com dois níveis
    moedas_th_manual_2_gray_img = cv2.inRange(moedas_gray_img, 110, 205)
    moedas_th_manual_2_gray_img = cv2.bitwise_not(moedas_th_manual_2_gray_img)
    cores_th_manual_2_gray_img = cv2.inRange(cores_gray_img, 140, 210)
    cores_th_manual_2_gray_img = cv2.bitwise_not(cores_th_manual_2_gray_img)
    # plot_gray(moedas_gray_img, moedas_th_manual_2_gray_img, 'Limiarizacao manual com 2 niveis')
    # plot_gray(cores_gray_img, cores_th_manual_2_gray_img, 'Limiarizacao manual com 2 niveis')

    ## Aplica transformacoes morfologicas nas imagens limearizadas
    # Moedas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    moedas_morph_img = moedas_th_manual_2_gray_img
    for i in xrange(5):
       moedas_morph_img = cv2.morphologyEx(moedas_morph_img, cv2.MORPH_OPEN, kernel)
       moedas_morph_img = cv2.dilate(moedas_morph_img,kernel,iterations=2)
       moedas_morph_img = cv2.morphologyEx(moedas_morph_img, cv2.MORPH_CLOSE, kernel)
    # plot_gray(moedas_gray_img, moedas_morph_img, 'Transformacoes morfologicas')
    # Cores
    kernel = np.ones((7,7),np.uint8)
    cores_morph_img = cores_th_manual_2_gray_img
    for i in xrange(20):
        cores_morph_img = cv2.morphologyEx(cores_morph_img, cv2.MORPH_OPEN, kernel)
    cores_morph_img = cv2.erode(cores_morph_img, kernel, iterations=1)
    # plot_gray(cores_gray_img, cores_morph_img, 'Transformacoes morfologicas')

    # Aplica o metodo de Otsu para limearizar as imagens
    moedas_otsu_retval, moedas_otsu_img = cv2.threshold(moedas_gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cores_otsu_retval, cores_otsu_img = cv2.threshold(cores_gray_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # plot_gray(moedas_gray_img, moedas_otsu_img, 'Otsu Threshold: %d' % moedas_otsu_retval)
    # plot_gray(cores_gray_img, cores_otsu_img, 'Otsu Threshold: %d' % cores_otsu_retval)

    #thresh = otsu_fast(np.array([[range(5),range(5),range(5),range(5),range(5)]], np.uint8), 3, 5)
    thresh = otsu_fast(cores_gray_img, 5, 256)
    #thresh = otsu_multi(cores_gray_img, 3, 256)
    print thresh, cores_otsu_retval
    th_img = apply_thresholds(cores_gray_img, list(thresh), thresh, 256)
    th_img = cv2.bitwise_not(th_img)
    plt.imshow(th_img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
