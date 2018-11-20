from PIL import Image
import scipy.misc
from scipy.misc import toimage, imsave
import math
import numpy
import numpy as np

#initialize the height and width of image
we = 256  #665
he = 256  #443

#initialize global numpy arrays used in the Canny Edge Detector
newgradientgx = np.zeros((he, we))
newgradientgy = np.zeros((he, we))
newgradientImage = np.zeros((he, we))
tan = np.zeros((he, we))
newtan = np.zeros((he, we))
p10= np.zeros((he, we))
p30= np.zeros((he, we))
p50= np.zeros((he, we))
gas= np.zeros((he, we))

#function for comparing the elements along the gradient direction for Non Maxima Suppression
def compare(a,b,c):
    if (a>b and a>c):
        return a
    else:
        return 0

#function to perform Gaussian blurring using the given mask
def gaussi2(a):
    #define mask
    mask = np.array([[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                     [1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0],
                     [2.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0],
                     [2.0, 4.0, 8.0, 16.0, 8.0, 4.0, 2.0],
                     [2.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0],
                     [1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0],
                     [1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]])
    gray_img = np.array(Image.open(a)).astype(np.uint8)
    h, w = gray_img.shape

   #offset each edge by 3 and apply filter
    for i in range(3, h - 3):
        for j in range(3, w - 3):
            gas[i,j] =  (mask[0, 0] * gray_img[i - 3, j - 3]) + \
                        (mask[0, 1] * gray_img[i - 3, j - 2]) + \
                        (mask[0, 2] * gray_img[i - 3, j - 1]) + \
                        (mask[0, 3] * gray_img[i - 3, j]) + \
                        (mask[0, 4] * gray_img[i - 3, j + 1]) + \
                        (mask[0, 5] * gray_img[i - 3, j + 2]) + \
                        (mask[0, 6] * gray_img[i - 3, j + 3]) + \
                        (mask[1, 0] * gray_img[i - 2, j - 3]) + \
                        (mask[1, 1] * gray_img[i - 2, j - 2]) + \
                        (mask[1, 2] * gray_img[i - 2, j - 1]) + \
                        (mask[1, 3] * gray_img[i - 2, j]) + \
                        (mask[1, 4] * gray_img[i - 2, j + 1]) + \
                        (mask[1, 5] * gray_img[i - 2, j + 2]) + \
                        (mask[1, 6] * gray_img[i - 2, j + 3]) + \
                        (mask[2, 0] * gray_img[i - 1, j - 3]) + \
                        (mask[2, 1] * gray_img[i - 1, j - 2]) + \
                        (mask[2, 2] * gray_img[i - 1, j - 1]) + \
                        (mask[2, 3] * gray_img[i - 1, j]) + \
                        (mask[2, 4] * gray_img[i - 1, j + 1]) + \
                        (mask[2, 5] * gray_img[i - 1, j + 2]) + \
                        (mask[2, 6] * gray_img[i - 1, j + 3]) + \
                        (mask[3, 0] * gray_img[i, j - 3]) + \
                        (mask[3, 1] * gray_img[i, j - 2]) + \
                        (mask[3, 2] * gray_img[i, j - 1]) + \
                        (mask[3, 3] * gray_img[i, j]) + \
                        (mask[3, 4] * gray_img[i, j + 1]) + \
                        (mask[3, 5] * gray_img[i, j + 2]) + \
                        (mask[3, 6] * gray_img[i, j + 3]) + \
                        (mask[4, 0] * gray_img[i + 1, j - 3]) + \
                        (mask[4, 1] * gray_img[i + 1, j - 2]) + \
                        (mask[4, 2] * gray_img[i + 1, j - 1]) + \
                        (mask[4, 3] * gray_img[i + 1, j]) + \
                        (mask[4, 4] * gray_img[i + 1, j + 1]) + \
                        (mask[4, 5] * gray_img[i + 1, j + 2]) + \
                        (mask[4, 6] * gray_img[i + 1, j + 3]) + \
                        (mask[5, 0] * gray_img[i + 2, j - 3]) + \
                        (mask[5, 1] * gray_img[i + 2, j - 2]) + \
                        (mask[5, 2] * gray_img[i + 2, j - 1]) + \
                        (mask[5, 3] * gray_img[i + 2, j]) + \
                        (mask[5, 4] * gray_img[i + 2, j + 1]) + \
                        (mask[5, 5] * gray_img[i + 2, j + 2]) + \
                        (mask[5, 6] * gray_img[i + 2, j + 3]) + \
                        (mask[6, 0] * gray_img[i + 3, j - 3]) + \
                        (mask[6, 1] * gray_img[i + 3, j - 2]) + \
                        (mask[6, 2] * gray_img[i + 3, j - 1]) + \
                        (mask[6, 3] * gray_img[i + 3, j]) + \
                        (mask[6, 4] * gray_img[i + 3, j + 1]) + \
                        (mask[6, 5] * gray_img[i + 3, j + 2]) + \
                        (mask[6, 6] * gray_img[i + 3, j + 3])

            #normalize by dividing by 140
            gas[i,j] = gas[i,j]/140
    toimage(gas).show()
    imsave('gaussian.bmp', gas)

#function to perform Prewitt operator
def prewitt(b):
    gray_img = np.array(Image.open(b)).astype(np.uint8)
    print("The values of the read image are ")
    print(gray_img)

    # Prewitt Operator
    h, w = gray_img.shape
    # define filters
    horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    #offset each edge by 1
    for i in range(5, h - 5):
        for j in range(5, w - 5):
            horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                             (horizontal[0, 1] * gray_img[i - 1, j]) + \
                             (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                             (horizontal[1, 0] * gray_img[i, j - 1]) + \
                             (horizontal[1, 1] * gray_img[i, j]) + \
                             (horizontal[1, 2] * gray_img[i, j + 1]) + \
                             (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                             (horizontal[2, 1] * gray_img[i + 1, j]) + \
                             (horizontal[2, 2] * gray_img[i + 1, j + 1])

            verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                           (vertical[0, 1] * gray_img[i - 1, j]) + \
                           (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                           (vertical[1, 0] * gray_img[i, j - 1]) + \
                           (vertical[1, 1] * gray_img[i, j]) + \
                           (vertical[1, 2] * gray_img[i, j + 1]) + \
                           (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                           (vertical[2, 1] * gray_img[i + 1, j]) + \
                           (vertical[2, 2] * gray_img[i + 1, j + 1])

            newgradientgx[i, j] = horizontalGrad
            newgradientgy[i, j] = verticalGrad

            if(newgradientgx[i,j]==0):
                tan[i,j]=90.00
            else:
                tan[i,j]=math.degrees(math.atan(newgradientgy[i,j]/newgradientgx[i,j]))
                if (tan[i,j]<0):
                    tan[i,j]= tan[i,j] + 360

            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag

#function to perform Non Maxima Suppression
def nonmaximasup(c):
    gray_img = np.array(Image.open(c)).astype(np.uint8)
    print("The read values for gradient are ")
    print(gray_img)
    h, w = gray_img.shape
    print('Max value is')
    print(numpy.amax(gray_img))

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (tan[i,j] > 157.5 and tan[i,j]<202.5) or (tan[i,j] > 0 and tan[i,j]<22.5) or (tan[i,j] > 337.5 and tan[i,j]<359.9):
                sector = 0
            elif (tan[i,j] > 22.5 and tan[i,j]<67.5) or (tan[i,j] > 202.5 and tan[i,j]<247.5):
                sector = 1
            elif (tan[i, j] > 247.5 and tan[i, j] < 292.5) or (tan[i, j] > 67.5 and tan[i, j] < 112.5):
                sector = 2
            else:
                sector = 3

            #Using the sector chart, we compare elements along the direction of gradient
            if (sector == 0):
                newtan[i,j] = compare (gray_img[i,j], gray_img[i,j-1], gray_img[i,j+1])
            elif (sector == 1):
                newtan[i,j] = compare (gray_img[i,j], gray_img[i-1,j+1], gray_img[i+1,j-1])
            elif (sector == 2):
                newtan[i,j] = compare (gray_img[i,j], gray_img[i-1,j], gray_img[i+1,j])
            elif (sector == 3):
                newtan[i, j] = compare(gray_img[i, j], gray_img[i - 1, j - 1], gray_img[i + 1, j + 1])

#function to perform p-tile thresholding
def ptile():
    k=0
    h=he-1
    w=we-1
    newtan2 = np.zeros((h+1) * (w+1))
    for i in range(0, h):
        for j in range(0,w):
            newtan2[k]=newtan[i][j]
            k=k+1
    newtan4=numpy.sort(np.ravel(newtan2))
    print(newtan4)

    #Consider only values above 0
    newtan3=np.trim_zeros(newtan4,'f')
    print("after removing 0: ")
    print(newtan3)

   #p-tile method for 10%
    print("For p-tile of 10%: ")
    threshold=newtan3[9*int((np.size(newtan3))/10)]
    threshold = newtan3[9 * int((np.size(newtan3)) / 10)]

    print("Threshold is ")
    print(threshold)
    count=0
    for i in range(0, h):
        for j in range(0, w):
            if(newtan[i,j]>threshold):
                p10[i,j] = newtan[i][j]
                count=count+1
            else:
                p10[i,j]=0
    print("Number of edge pixels is ")
    print(count)
    print(" ")
    toimage(p10).show()
    imsave('ptile10.bmp', p10.astype(np.uint8))


    # p-tile method for 30%
    print("For p-tile of 30%: ")
    threshold=newtan3[7*int((np.size(newtan3))/10)]
    print("Threshold is ")
    print(threshold)
    count = 0
    for i in range(0, h):
        for j in range(0, w):
            if (newtan[i, j] > threshold):
                p30[i, j] = newtan[i][j]
                count = count + 1
            else:
                p30[i, j] = 0
    print("Number of edge pixels is ")
    print(count)
    print(" ")
    toimage(p30).show()
    imsave('ptile30.bmp', p30.astype(np.uint8))


    # p-tile method for 50%
    print("For p-tile of 50%: ")
    threshold=newtan3[5*int((np.size(newtan3))/10)]
    print("Threshold is ")
    print(threshold)
    count = 0
    for i in range(0, h):
        for j in range(0, w):
            if (newtan[i, j] > threshold):
                p50[i, j] = newtan[i][j]
                count = count + 1
            else:
                p50[i, j] = 0
    print("Number of edge pixels is ")
    print(count)
    toimage(p50).show()
    imsave('ptile50.bmp', p50.astype(np.uint8))


#Driver Program
indimage = scipy.misc.imread("Lena256.bmp", flatten=True)
numpy.savetxt('raw_values.txt',indimage, delimiter=',', fmt='%i')
print(indimage.shape)
print(indimage)

#perform Gaussian blurring with the given mask
gaussi2('Lena256.bmp')

#perform Gradient Operation using Prewitt operator
prewitt('gaussian.bmp')

toimage(newgradientgx).show()
toimage(newgradientgy).show()
toimage(newgradientImage).show()
numpy.savetxt('gradient255.txt',newgradientImage, delimiter=',', fmt='%i')



imsave('xgradient.bmp', newgradientgx)
imsave('ygradient.bmp', newgradientgy)
imsave('magnitude.bmp', newgradientImage)

print("Vertical Gradient values are ")
print(newgradientgx)
print(" ")
print("Horizontal Gradient values are ")
print(newgradientgy)
print(" ")
print("Gradient values are ")
print(newgradientImage)
print(" ")
print("Tan values are ")
print(tan)
print(" ")

#Perform Non Maxima Suppression
nonmaximasup('magnitude.bmp')
toimage(newtan).show()
imsave('nonmaximasuppression.bmp', newtan.astype(np.uint8))
print(numpy.amax(newtan))

#Perform p-tile thresholding
ptile()