import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import cv2
from matplotlib.pyplot import figure

def median(img_arr, stride):
    output = np.zeros(img_arr.shape)
    
    x,y = img_arr.shape

    for i in range(stride//2,x-stride//2):
        for j in range(stride//2,y-stride//2):
            arr_slice = img_arr[i-stride//2:i+stride//2,j-stride//2:j+stride//2]
            
            output[i,j] = np.median(arr_slice)
    return output

def mean(img_arr, stride):
    output = np.zeros(img_arr.shape)
    
    x,y = img_arr.shape

    for i in range(stride//2,x-stride//2):
        for j in range(stride//2,y-stride//2):
            arr_slice = img_arr[i-stride//2:i+stride//2,j-stride//2:j+stride//2]
            
            output[i,j] = np.mean(arr_slice)
    return output

def conv2d(img_arr,filtro):
    output = np.zeros(img_arr.shape)

    x,y = img_arr.shape

    x_stride,y_stride = filtro.shape

    for i in range(x_stride//2,x-ceil(x_stride/2)):
        for j in range(y_stride//2,y-ceil(y_stride/2)):
            arr_slice = img_arr[i-x_stride//2:i+ceil(x_stride/2),j-y_stride//2:j+ceil(y_stride/2)]
            
            output[i,j] = np.sum(arr_slice*filtro)
    
    return output
    
def sobel(img_arr):
    horizontal_filter = np.array([1,0,-1,2,0,-2,1,0,-1]).reshape(3,3)
    vertical_filter   = np.array([1,2,1,0,0,0,-1,-2,-1]).reshape(3,3)
    
    x_result = conv2d(img_arr,horizontal_filter)
    y_result = conv2d(img_arr,vertical_filter)
    
    result = x_result + y_result
    
    return result

def gamma_correction(img_arr,gamma):
    output = np.zeros(img_arr.shape)
    
    gamma_correction = img_arr.flatten().astype(np.uint16)
    
    output = np.asarray([((x/255)**gamma)*255 for x in gamma_correction]).astype(np.uint8)
    output = output.reshape(img_arr.shape)
    
    return output

def histogram(img_arr):
    flat = img_arr.flatten()
    hist = np.zeros(256,dtype=np.uint64)

    # para evitar usar o np.bincount():
    for i in range(0,256):
        hist[i] = np.count_nonzero(flat == i)
        
    hist_acc_prob = np.zeros(256)
    num_pixels = len(flat)
    hist_prob = hist/num_pixels

    hist_acc_prob = mask(hist_prob)
    
    transformation_map = np.ceil(255 * hist_acc_prob).astype(np.uint8)

    new_img_flat = np.array([transformation_map[x] for x in flat])
    new_img_arr = new_img_flat.reshape(img_arr.shape)

    new_hist = np.zeros(256)
    new_flat = new_img_arr.flatten()
    for i in range(0,256):
        new_hist[i] = np.count_nonzero(new_flat == i)

    f, axarr = plt.subplots(1,2,figsize=(15,5))
    f.suptitle('Histogram Normalization')
    axarr[0].plot(hist)
    axarr[0].title.set_text('before')
    axarr[1].plot(new_hist)
    axarr[1].title.set_text('after')

    
    return new_img_arr

def mask(arr):
    sum_arr = [0]
    sum_arr[0] = arr[0]
    for i in range(len(arr)):
        if i != 0:
            sum_arr.append(arr[i] + sum_arr[i - 1])

    return np.asanyarray(sum_arr)

def gaussian(img_arr):
    gaussian_filter = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]).reshape(5,5)/273
    
    return conv2d(img_arr,gaussian_filter)
    
def mediana(image, rgb, padding = 5):
    
    img_numpy = np.asarray(image)
    R = img_numpy[: ,:,rgb].copy()
    color_image = img_numpy.copy()
    
    result = np.zeros(R.shape)

    #fitro de mediana
    for i in range(padding//2,640-padding//2):
        for j in range(padding//2,960-padding//2):
            arr_slice = R[i-padding//2:i+padding//2,j-padding//2:j+padding//2]
            
            result[i,j] = np.median(arr_slice)
    
    color_image[: ,:,rgb] = result
    return color_image


def usando_laplace(image_orig):
    # Here we define the matrices associated with the Sobel filter
    filter1 = np.array([[0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0]])
    filter = np.array([[1.0, 1.0, 1.0],
                    [1.0, -8.0, 1.0],
                    [1.0, 1.0, 1.0]])
    
    blur_image = cv2.GaussianBlur(image_orig.copy(), (3, 3), 0)  # we need to know the shape of the input grayscale image
    laplace_filtered_image = cv2.filter2D(blur_image,-1,filter)  # initialization of the output image array (all elements are 0)
    
    return laplace_filtered_image

def gaussian_filter(I_shape, filter_params):
        P = I_shape[0]/2
        Q = I_shape[1]/2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
        Duv = (((U-P)**2+(V-Q)**2)).astype(float)
        H = np.exp((-Duv/(2*(filter_params[0])**2)))
        return (1 - H)
def homo_gaussiano(img,a,b,rgb):
    
    img_numpy = np.asarray(img)
    color_image = img_numpy.copy()
    R = img_numpy[: ,:,rgb].copy()
    
    I_log = np.log1p(R)
    I_fft = np.fft.fft2(I_log)

    # Implementando filtro gaussiano 
     
    H = gaussian_filter(I_shape = I_fft.shape,filter_params=[10,2])
    
    # Apply filter on frequency domain then take the image back to spatial domain
    #I_fft_filt = self.__apply_filter(I = I_fft, H = H)
    H = np.fft.fftshift(H)
    I_fft_filt = (a +b*H)*I_fft
    
    I_filt = np.fft.ifft2(I_fft_filt)
    I = np.exp(np.real(I_filt))-1
    result= np.uint8(I)
   #result = Image.filter(ImageFilter.GaussianBlur)


    figure(figsize=(11,7))
    plt.imshow(result, cmap='gray')
    
    plt.show()

    color_image[: ,:,rgb] = result
    return color_image

'''
fig = plt.figure(figsize=(20, 15))

for i in range(3):
    img = image_array[:,:,i]
    fig.add_subplot(1,3,i+1)
    plt.imshow(img,cmap='gray')
    plt.axis('off')
plt.show()
'''