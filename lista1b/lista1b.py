import cv2
from matplotlib import pyplot as plt
import numpy as np
import itertools

PATH = ['moedas.jpg', 'cores.jpeg']

def otsu_fast(img, numcl, maxval):
    """Implementacao de Ping-Sung Liao et. al.,
       'A Fast Algorithm for Multilevel Thresholding'
       doi=10.1.1.85.3669
    """
    # Constroi o histograma
    hist = np.bincount(img.ravel(),minlength=maxval)
    hist = np.float64(hist)/np.sum(hist)
    # Aloca matrizes auxiliares
    P = np.zeros((maxval,maxval))
    S = np.zeros((maxval,maxval))
    H = np.zeros((maxval,maxval))
    # Calcula matrizes auxiliares
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

    # Determina thresholds otimos t* = (t*1,..t*n)
    # maximizando a funcao de variancia alternativa
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

def apply_thresholds(img, thresh, maxval):
    thresh.insert(0, 0)
    thresh.append(maxval)
    h, w = img.shape
    retimg = np.ndarray((h,w), np.uint8)
    for y in xrange(h):
        for x in xrange(w):
            for i in xrange(len(thresh)-1):
                if img[y,x] > thresh[i] and img[y,x] <= thresh[i+1]:
                    retimg[y,x] = thresh[i]
    return retimg

def make_mask(img, maxval, delta=3):
    hist = np.bincount(img.ravel(),minlength=maxval)
    peak = np.argmax(hist)
    h, w = img.shape
    retimg = np.ndarray((h,w), np.uint8)
    for y in xrange(h):
        for x in xrange(w):
            retimg[y,x] = 0 if (img[y,x] > peak-delta and img[y,x] < peak+delta) else 255
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
    retval, moedas_th_manual_gray_img = cv2.threshold(moedas_gray_img, 116, 255,
        cv2.THRESH_BINARY_INV)
    retval, cores_th_manual_gray_img = cv2.threshold(cores_gray_img, 140, 255,
        cv2.THRESH_BINARY_INV)
    # plot_gray(moedas_gray_img, moedas_th_manual_gray_img, 'Limiarizacao manual com 1 nivel')
    # plot_gray(cores_gray_img, cores_th_manual_gray_img, 'Limiarizacao manual com 1 nivel')
    cv2.imwrite('resultados/moedas_th_manual_gray_img.jpg', moedas_th_manual_gray_img)
    cv2.imwrite('resultados/cores_th_manual_gray_img.jpg', cores_th_manual_gray_img)

    # Limiariza a imagem com dois níveis
    moedas_th_manual_2_gray_img = cv2.inRange(moedas_gray_img, 110, 205)
    moedas_th_manual_2_gray_img = cv2.bitwise_not(moedas_th_manual_2_gray_img)
    cores_th_manual_2_gray_img = cv2.inRange(cores_gray_img, 140, 210)
    cores_th_manual_2_gray_img = cv2.bitwise_not(cores_th_manual_2_gray_img)
    # plot_gray(moedas_gray_img, moedas_th_manual_2_gray_img, 'Limiarizacao manual com 2 niveis')
    # plot_gray(cores_gray_img, cores_th_manual_2_gray_img, 'Limiarizacao manual com 2 niveis')
    cv2.imwrite('resultados/moedas_th_manual_2_gray_img.jpg', moedas_th_manual_2_gray_img)
    cv2.imwrite('resultados/cores_th_manual_2_gray_img.jpg', cores_th_manual_2_gray_img)

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
    cv2.imwrite('resultados/moedas_morph_img.jpg', moedas_morph_img)
    cv2.imwrite('resultados/cores_morph_img.jpg', cores_morph_img)

    # Aplica o metodo de Otsu para limearizar as imagens
    moedas_otsu_retval, moedas_otsu_img = cv2.threshold(moedas_gray_img, 0, 255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cores_otsu_retval, cores_otsu_img = cv2.threshold(cores_gray_img, 0, 255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # plot_gray(moedas_gray_img, moedas_otsu_img, 'Otsu Threshold: %d' % moedas_otsu_retval)
    # plot_gray(cores_gray_img, cores_otsu_img, 'Otsu Threshold: %d' % cores_otsu_retval)
    cv2.imwrite('resultados/moedas_otsu_img.jpg', moedas_otsu_img)
    cv2.imwrite('resultados/cores_otsu_img.jpg', cores_otsu_img)

    # Metodo de Otsu para multiplas classes
    moedas_multi_otsu_th = otsu_fast(moedas_gray_img, 3, 256)
    cores_multi_otsu_th = otsu_fast(cores_gray_img, 3, 256)
    moedas_multi_otsu_img = apply_thresholds(moedas_gray_img,
        list(moedas_multi_otsu_th), 256)
    cores_multi_otsu_img = apply_thresholds(cores_gray_img,
        list(cores_multi_otsu_th), 256)
    # plot_gray(moedas_gray_img, moedas_multi_otsu_img,
    #     'Multi-otsu:'+str(moedas_multi_otsu_th))
    # plot_gray(cores_gray_img, cores_multi_otsu_img,
    #     'Multi-otsu:'+str(cores_multi_otsu_th))
    cv2.imwrite('resultados/moedas_multi_otsu_img.jpg', moedas_multi_otsu_img)
    cv2.imwrite('resultados/cores_multi_otsu_img.jpg', cores_multi_otsu_img)

    ## Metodo de Otsu aplicado no Hue
    # Aplicacao de filtro gaussiano
    moedas_img = cv2.GaussianBlur(moedas_img,(15,15),0)
    # cores_img = cv2.GaussianBlur(cores_img,(15,15),0)
    # Conversao de BGR para HSV
    moedas_hsv_img = cv2.cvtColor(moedas_img, cv2.COLOR_BGR2HSV)
    cores_hsv_img = cv2.cvtColor(cores_img, cv2.COLOR_BGR2HSV)
    moedas_h_ch, moedas_s_ch, moedas_v_ch = cv2.split(moedas_hsv_img)
    cores_h_ch, cores_s_ch, cores_v_ch = cv2.split(cores_hsv_img)
    # Multi Otsi no canal da Matiz (Hue)
    moedas_h_ch_th = otsu_fast(moedas_h_ch, 5, 180)
    cores_h_ch_th = otsu_fast(cores_h_ch, 5, 180)
    moedas_mask = make_mask(moedas_h_ch, 180, delta=5)
    cores_mask = make_mask(cores_h_ch, 180)
    moedas_seg = cv2.bitwise_and(moedas_img, moedas_img, mask=moedas_mask)
    cores_seg = cv2.bitwise_and(cores_img, cores_img, mask=cores_mask)
    cv2.imwrite('resultados/cores_seg.jpg', cores_seg)
    cv2.imwrite('resultados/moedas_seg.jpg', moedas_seg)
    moedas_h_ch = apply_thresholds(moedas_h_ch, list(moedas_h_ch_th), 180)
    cores_h_ch = apply_thresholds(cores_h_ch, list(cores_h_ch_th), 180)
    cv2.imwrite('resultados/moedas_multi_otsu_hue_img.jpg', moedas_h_ch)
    cv2.imwrite('resultados/cores_multi_otsu_hue_img.jpg', cores_h_ch)

if __name__ == '__main__':
    main()
