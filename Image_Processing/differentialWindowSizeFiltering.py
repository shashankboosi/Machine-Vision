# The logic to the code is written by sboosi
import cv2
import numpy as np

# Read the image
inputImage = cv2.imread('../Images/Oil-Paint/rail/grayImageRailOutput.jpg', cv2.IMREAD_GRAYSCALE)

print(inputImage.shape)
if inputImage is None:
    exit('Please check the file location correctly!')

# Display the image
cv2.imshow("Filtering the image", inputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use 3 window sizes
windowSize = [[5, 5], [15, 15], [25, 25]]
for size in windowSize:
    # Calculate the border sizes to be padded in all the directions
    print(size[0])
    border = int(size[0] / 2)
    paddedImage = cv2.copyMakeBorder(inputImage, border, border, border, border, cv2.BORDER_REFLECT_101)
    outputImage = np.copy(inputImage)
    # Operation to use the window sizes and perform filtering
    for i in range(inputImage.shape[0]):
        print(i)
        for j in range(inputImage.shape[1]):
            windowImage = paddedImage[i: i + size[0], j: j + size[1]]
            histMax = np.argmax(cv2.calcHist([windowImage], [0], None, [256], [0, 256]))
            outputImage[i][j] = histMax

    # Store the images by writing it the directory
    file_extensions = '../OutputImages/Oil-Paint/rail/railFilterWithSize{}.jpg'.format(size[0])
    cv2.imwrite(file_extensions, outputImage)
