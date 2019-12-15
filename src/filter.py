############################################################
# python::support libraries
############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal, misc, ndimage
from PIL import Image, ImageFilter


############################################################
# __filter::low pass filter to denoise high frequencies
############################################################
def __filter(FFT_pixels):

	# Define the fraction of coefficients (in each direction) we keep
    keep_fraction = 0.09
    fft2 = FFT_pixels.copy()

    # Set r and c to be the number of rows and columns of the array.
    r, c = fft2.shape

    # Set to zero all rows with indices between r*keep_fraction and r*(1-keep_fraction)
    fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

    # Similarly with the columns
    fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

    # plot data after applying filter
    plt.figure()
    __spectrum(fft2)
    plt.title('Filtered Spectrum')
    plt.show()

    # Apply inverse FFT to restore the image
    restored = fftpack.ifft2(fft2).real
    return restored


############################################################
# __readImage::read image and return pixels 2D array
############################################################
def __readImage(imageName):
	pixels = plt.imread(imageName).astype(float)
	plt.figure()
	plt.imshow(pixels, plt.cm.gray)
	plt.title('Original Image')
	plt.show()
	return pixels

############################################################
# __spectrum::plots the FFT spectrum
############################################################
def __spectrum(FFT_pixels):
   from matplotlib.colors import LogNorm
   # A logarithmic colormap
   plt.imshow(np.abs(FFT_pixels), norm=LogNorm(vmin=5))
   plt.colorbar()

############################################################
# main:: read halftone image and apply denoise filter
############################################################
print('============================================================')
print('               DSP:Halftone Image Enhancement')
print('============================================================')
# Read Image Name From User
imageName = input('Image Name(Path):: ')

# Open the image and plot it 
imagePixels = __readImage(imageName)


# Apply 2D-FFT and plot it
FFT_pixels = fftpack.fft2(imagePixels)
plt.figure()
__spectrum(FFT_pixels)
plt.title('Fourier Transform')
plt.show()


# Apply Low Pass Filter
restored= __filter(FFT_pixels)

# Apply gaussian filter to enhance the image more
blurred_f = ndimage.gaussian_filter(restored, 2)

filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

alpha = 30

# Sharpen the image
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

misc.imsave('restored-'+imageName, blurred_f) 
im = Image.open('restored-'+imageName)
edgeEnahnced = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
smoothed = edgeEnahnced.filter(ImageFilter.SMOOTH_MORE)

# plot the two images after low pass filter and after gaussian filter
plt.figure(figsize=(16, 4))
plt.subplot(131)
plt.imshow(restored, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Restored Image')
plt.subplot(132)
plt.imshow(blurred_f, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Gaussian Filter')
plt.subplot(133)
plt.imshow(smoothed, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.title('Enhanced Image')
plt.show()

# save the moothed image
misc.imsave('enhanced-'+imageName, smoothed) 


# End of Project
print('============================================================')
print('                          DSP::End')
print('============================================================')
