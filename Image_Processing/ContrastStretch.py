import cv2
import numpy as np

image = cv2.imread("../Images/ansel_adams.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

originalImagePixels = np.array(gray)
print(originalImagePixels.shape)

cv2.imshow('Original image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Input parameters
a = 0
b = 255
c = originalImagePixels.min()
d = originalImagePixels.max()

# Contrast Stretching Formula
for x in range(0, originalImagePixels.shape[0]):
    for y in range(0, originalImagePixels.shape[1]):
        originalImagePixels[x][y] = (originalImagePixels[x][y] - c) * ((b - a) / (d - c)) + a

cv2.imshow('Contrast image', originalImagePixels)
cv2.waitKey(0)
cv2.destroyAllWindows()

LaplacianApproximateKernel = np.array([[0, -1, 0],
                                       [-1, 4, -1],
                                       [0, -1, 0]], dtype=int)

finalConvolvedImage = cv2.filter2D(originalImagePixels, -1, LaplacianApproximateKernel)

cv2.imshow('Final image', finalConvolvedImage)
cv2.imwrite('../OutputImages/ConvolvedImage.jpg', finalConvolvedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
