# The logic to the code is written by sboosi
from matplotlib import pyplot as plt
import cv2

dogInputFiles = ['../OutputImages/Oil-Paint/dog/dogFilterWithSize5.jpg',
                 '../OutputImages/Oil-Paint/dog/dogFilterWithSize15.jpg',
                 '../OutputImages/Oil-Paint/dog/dogFilterWithSize25.jpg']

outputFiles = '../OutputImages/Oil-Paint/plots/combinedGrayDogImageWithAll3Filters.png'

# Show the images of all the dog filters used for oil-paint filtering
for i in range(len(dogInputFiles)):
    image = cv2.imread(dogInputFiles[i])
    plt.subplot(1, 3, i + 1)
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.savefig(outputFiles, bbox_inches='tight')
