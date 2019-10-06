import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
inputOriginalImage = cv2.imread('../Images/Oil-Paint/Flower/flower.jpg', cv2.IMREAD_COLOR)
if inputOriginalImage is None:
    exit('Please check the file location correctly!')
# convert BGR to RGB
inputOriginalImage = cv2.cvtColor(inputOriginalImage, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(inputOriginalImage)

# Display the image
cv2.imshow("Original Image", inputOriginalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Read the gray scale image
inputGreyScaleFilteredImage = cv2.imread('../Images/Oil-Paint/Flower/flowerFilterWithSize3.png', cv2.IMREAD_GRAYSCALE)
if inputGreyScaleFilteredImage is None:
    exit('Please check the file location correctly!')

# Display the image
cv2.imshow("Gray Scale Image", inputGreyScaleFilteredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

window_name = "Filtering the image"

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# Display the image
cv2.imshow("Filtering the image", inputGreyScaleFilteredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()


def make_border(image, borderName):
    return cv2.copyMakeBorder(image, borderName, borderName, borderName, borderName,
                              cv2.BORDER_CONSTANT, value=0)


# Use 3 window sizes which are of odd size
windowSize = [[3, 3], [5, 5], [7, 7]]
for size in windowSize:
    # Calculate the border sizes to be padded in all the directions
    border = int(size[0] / 2)
    outputRImage = np.copy(r)
    outputGImage = np.copy(g)
    outputBImage = np.copy(b)
    # Using zero padding as the zeros does not affect the algorithm
    paddedOriginalRChannelImage = make_border(r, border)
    paddedOriginalGChannelImage = make_border(g, border)
    paddedOriginalBChannelImage = make_border(b, border)

    paddedGrayImage = make_border(inputGreyScaleFilteredImage, border)
    # Operation to use the window sizes and perform filtering
    for i in range(inputGreyScaleFilteredImage.shape[0]):
        for j in range(inputGreyScaleFilteredImage.shape[1]):

            windowGrayImage = paddedGrayImage[i: i + size[0], j: j + size[1]]
            center = windowGrayImage[border, border]
            RIntensities = 0
            BIntensities = 0
            GIntensities = 0
            intensityCount = 0
            for row in range(windowGrayImage.shape[0]):
                for col in range(windowGrayImage.shape[1]):
                    if row == border and col == border:
                        continue
                    if center == windowGrayImage[row][col]:
                        RIntensities = RIntensities + paddedOriginalRChannelImage[i + row][j + col]
                        GIntensities = GIntensities + paddedOriginalGChannelImage[i + row][j + col]
                        BIntensities = BIntensities + paddedOriginalBChannelImage[i + row][j + col]
                        intensityCount += 1
            if intensityCount != 0:
                outputRImage[i][j] = RIntensities / intensityCount
                outputGImage[i][j] = GIntensities / intensityCount
                outputBImage[i][j] = BIntensities / intensityCount

    # Merge all the channels to form an RGB image
    mergedOutputImage = cv2.merge((outputRImage, outputGImage, outputBImage))
    # Store the images by writing it the directory
    file_extensions = '../OutputImages/Oil-Paint/Flower/flowerOilPaintForWindow3WithSize{}.png'.format(size[0])
    cv2.imwrite(file_extensions, mergedOutputImage)

# # show original and filtered images
# plt.subplot(1, 3, 1), plt.imshow(inputOriginalImage), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 2), plt.imshow(blur), plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 3), plt.imshow(blur2), plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
