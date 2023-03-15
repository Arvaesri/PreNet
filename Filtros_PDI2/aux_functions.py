from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.pyplot import figure

def load_image(file_path,color_model = 'RGB'):
    image = Image.open(file_path)
    image = image.convert(color_model)
    return image

def normalizeAndShowImage(image_array):
    image_array = image_array + abs(image_array.min())
    image_array = (image_array * 255)/image_array.max()
    try:
        final_image = Image.fromarray(image_array.astype(np.uint8),'RGB')
    except ValueError:
        final_image = Image.fromarray(image_array.astype(np.uint8))
    display(final_image)
    
def plot_col_hist(img):
    height, width, channels = img.shape
    colors = ('b', 'g', 'r')
    features=[]


    #Exemplo PYImageSearch
    #Carrega o Histograma da imagem inteira
    chans = cv2.split(img)
    for (chan, color) in zip(chans, colors):
        hist_full = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist_full)
        plt.plot(hist_full, color=color)
        plt.xlim([0, 256])
    plt.show()

def plotando(image_orig,image_filt,dir_save):
    
    '''figure(figsize=(11,7))
    plt.imshow(image_orig)
    plt.show()
    figure(figsize=(11,7))
    plt.imshow(image_filt)
    plt.show()'''
    
    im_rgb = cv2.cvtColor(np.asarray(image_orig.copy()), cv2.COLOR_BGR2RGB)
    imf_rgb = cv2.cvtColor(image_filt.copy(), cv2.COLOR_BGR2RGB)

    cv2.imwrite(dir_save,imf_rgb)
    
    '''plot_col_hist(np.asarray(im_rgb.copy()))
    plot_col_hist(np.asarray(imf_rgb.copy()))'''

def plotRGBChannels(img_arr):
    R, G, B = [ img_arr[:,:,x] for x in range(3)]
    red_display = np.zeros(img_arr.shape,dtype=np.uint8)
    green_display = np.zeros(img_arr.shape,dtype=np.uint8)
    blue_display = np.zeros(img_arr.shape,dtype=np.uint8)
    red_display[:,:,0] = R
    green_display[:,:,1] = G
    blue_display[:,:,2] = B
    f, axarr = plt.subplots(1,3,figsize=(21,20))
    axarr[0].imshow(red_display)
    axarr[0].set_title('RED')
    axarr[1].imshow(green_display)
    axarr[1].set_title('GREEN')
    axarr[2].imshow(blue_display)
    axarr[2].set_title('BLUE')
    for ax in axarr.flat:
        ax.set_xticks([])
        ax.set_yticks([])
