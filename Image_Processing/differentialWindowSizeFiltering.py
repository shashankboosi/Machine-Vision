import cv2
import numpy as np
from matplotlib import pyplot as plt

inputImage = cv2.imread('../Images/Oil-Paint/grayImageFlower.png', cv2.IMREAD_GRAYSCALE)
window_name = "Filtering the image"

if inputImage is None:
    exit('Please check the file location correctly!')

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# Display the image
cv2.imshow("Filtering the image", inputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

windowSize = [[3, 3], [5, 5], [7, 7]]
for size in windowSize:
    # Window size windowSize[i][0] and windowSize[i][1]
    border = int(size[0] / 2)
    outputImage = np.copy(inputImage)
    for i in range(inputImage.shape[0]):
        for j in range(inputImage.shape[1]):
            paddedImage = cv2.copyMakeBorder(inputImage, border, border, border, border, cv2.BORDER_REFLECT_101)
            windowImage = paddedImage[i: i + size[0], j: j + size[1]]
            hist = np.argmax(np.ravel((cv2.calcHist([windowImage], [0], None, [256], [0, 256]))))
            outputImage[i][j] = hist

    file_extensions = '../OutputImages/Oil-Paint/flowerFilterWithSize{}.png'.format(size[0])
    cv2.imwrite(file_extensions, outputImage)
