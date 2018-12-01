![test](https://raw.githubusercontent.com/dnezan/dudemen-the-game/master/img/l.gif)

# Canny's Edge Detector
This is an implementation of Canny's Edge Detector which does not use prebuilt libraries for 
The Canny’s Edge Detector consists of four steps: Gaussian smoothing, gradient operation,
non-maxima suppression and thresholding. The input to your program is a grayscale image of
size N X M. Use the 7 x 7 Gaussian mask as shown below (on page 2) for smoothing the input
image. Use the center of the mask as the reference center. If part of the Gaussian mask goes outside of
the image border, let the output image be undefined at the reference center’s location. Note that the
sum of the entries in the Gaussian mask do not sum to 1. After performing convolution (or crosscorrelation),
you need to perform normalization by dividing the result by the sum of the entries
(= 140 for the given mask) at each pixel location. Instead of using the Robert’s operator, use the
Prewitt’s operator for gradient operation. If part of the 3 x 3 masks of the Prewitt’s operator lies in
the undefined region of the image after Gaussian filtering, set the output value to zero (indicates
no edge). Use simple thresholding in the fourth step but use the P-tile method to set the threshold T
(described below.)

#How to compile and run the program
The only libraries used in this program is PIL and scipy in order to read and write the image, numpy in order to save the 0-255 value of each pixel location, and math to compute the square root. No other libraries or in built functions are required for any operation including convolution. 

