# Canny's Edge Detector
This is an implementation of Canny's Edge Detector which does not use prebuilt libraries for 
The Canny’s Edge Detector consists of four steps:  
• Gaussian smoothing  
• Gradient operation  
• Non-maxima suppression  
• Thresholding  

The input to the program is a grayscale image of
size N X M.  
![gaussian](https://raw.githubusercontent.com/dnezan/cannyedgedetector/master/Lena256.bmp)

Instead of using the Robert’s operator, use the
Prewitt’s operator for gradient operation. If part of the 3 x 3 masks of the Prewitt’s operator lies in
the undefined region of the image after Gaussian filtering, set the output value to zero (indicates
no edge). Use simple thresholding in the fourth step but use the P-tile method to set the threshold T
(described below.)

## How to compile and run the program
The only libraries used in this program is PIL and scipy in order to read and write the image, numpy in order to save the 0-255 value of each pixel location, and math to compute the square root. No other libraries or in built functions are required for any operation including convolution. 

## Functions
### Gaussian Smoothening
The 7 x 7 Gaussian mask as shown below is used for smoothing the input
image. The center of the mask is used as the reference center. If part of the Gaussian mask goes outside of
the image border, the output pixel is set to 0. Note that the
sum of the entries in the Gaussian mask do not sum to 1. After performing convolution (or crosscorrelation),
normalization is performed by dividing the result by the sum of the entries
(= 140 for the given mask) at each pixel location.  
![gaussian](https://raw.githubusercontent.com/dnezan/cannyedgedetector/master/img/mask.png)    

### Gradient Operation
The Prewitt’s operator is used for gradient operation. If part of the 3 x 3 masks of the Prewitt’s operator lies in
the undefined region of the image after Gaussian filtering, output value is set to zero (indicates
no edge).  

### Non-Maxima Suppression  
Non-maximum suppression (NMS)is a post-processing algorithm responsible for merging all detections that belong to the same object.  

### Thresholding  
Simple thresholding is performed in the fourth step but uses the P-tile method to set the threshold T for P = 10%, 30% and 50%.

