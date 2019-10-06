import numpy as np
import cv2

# Read an image
inputImage = cv2.imread('../Images/Oil-Paint/Flower/flower.jpg', cv2.IMREAD_COLOR)

if inputImage is None:
    exit('Please check the file location correctly!')

# Display the image
cv2.imshow('Original Image', inputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print image dimension
print(inputImage.shape)

# Splitting the image to B, G, R
b, g, r = cv2.split(inputImage)
print(b.shape)

# Initialize the GrayImage pixels to the dimensions of the original image
grayImage = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.uint8)

# Create a new image from the original image using the transformation equation
for i in range(inputImage.shape[0]):
    for j in range(inputImage.shape[1]):
        grayImage[i][j] = ((0.299 * r[i][j]) + (0.587 * g[i][j]) + (0.114 * b[i][j]))

# Rounding the pixels to display the output image
outputGrayImage = np.round_(grayImage, decimals=0)

# Store the image by writing it to the images directory
cv2.imwrite('../OutputImages/Oil-Paint/Flower/grayImageFlowerOutput.png', outputGrayImage)
