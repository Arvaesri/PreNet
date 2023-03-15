'''
Escola Politécnica de Pernambuco
@Prof: Bruno Fernandes
@authors: Thiago Porfirio 
'''

from re import I
from PIL import Image, ImageFilter
import numpy as np
from math import ceil
import matplotlib.pyplot as plt 
from skimage import data
import math
from matplotlib.pyplot import figure
import cv2
import os
import scipy
from scipy import ndimage
from aux_functions import load_image, normalizeAndShowImage, plotRGBChannels, plot_col_hist, plotando
from filtros import mean, median, conv2d, sobel, histogram, gamma_correction, gaussian, mediana, usando_laplace, gaussian_filter, homo_gaussiano
from arvore import Node
import re



def main():
    


    for i in os.listdir('Imagens/'):
        print(i)

        agucar = Image.open('Imagens/'+i)
        #plotRGBChannels(np.asarray(agucar))
        agucar_lap = usando_laplace(np.asarray(agucar))
        
        teste=cv2.subtract(np.asarray(agucar),agucar_lap.copy())

        blur_agucar = cv2.GaussianBlur(np.asarray(agucar), (3, 3), 0)
        detalhe = cv2.subtract(np.asarray(agucar),blur_agucar)

        agucar_trat=cv2.add(teste.copy(),detalhe.copy())
        
        plotando(agucar,agucar_trat,dir_save="Resultados/"+i)

        #5 Melhorar imagem
        guarda = Image.open('Imagens/'+i)
        
        color_image = guarda
        #color_image= mediana(guarda.copy(),0,5)
        #color_image= cv2.medianBlur(guarda,3)
        blur_agucar = cv2.GaussianBlur(np.asarray(color_image.copy()), (3, 3), 0)
        detalhe = cv2.subtract(np.asarray(color_image),blur_agucar)

        guarda_filt=cv2.add(np.asarray(color_image.copy()),detalhe.copy())
        guarda_filt=cv2.add(np.asarray(color_image.copy()),detalhe.copy())

        plotando(guarda,guarda_filt,dir_save="Resultados2/"+i)
        
    


    

    


    


main()



