import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image

size = 100, 100

img_names = ["../Images/Segmentation/strawberry.png", "../Images/Segmentation/shapes.png"]
ext_names = ["../Images/Segmentation/coins.png", "../Images/Segmentation/two_halves.png"]
output_path_extension = '../OutputImages/Segmentation/'

images = [i for i in img_names]
ext_images = [i for i in ext_names]


def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3, output_path):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.show()
    fig.savefig(output_path)


for img_path in images:
    img = Image.open(img_path)
    img.thumbnail(size)  # Convert the image to 100 x 100
    # Convert the image to a numpy matrix
    img_mat = np.array(img)[:, :, :3]

    # --------------- Mean Shift algortithm ---------------------

    # Extract the three RGB colour channels
    b, g, r = cv2.split(img_mat)

    # Combine the three colour channels by flatten each channel
    # then stacking the flattened channels together.
    # This gives the "colour_samples"
    colour_samples = np.stack((b.flatten(), g.flatten(), r.flatten()), axis=1)

    # Perform Meanshift  clustering
    ms_clf = MeanShift(bin_seeding=True)
    ms_labels = ms_clf.fit_predict(colour_samples)

    # Reshape ms_labels back to the original image shape for displaying the segmentation output
    ms_labels = np.reshape(ms_labels, b.shape)

    # ------------- Water Shed algortithm --------------------------

    # Convert the image to gray scale and convert the image to a numpy matrix
    img_array = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)

    # Calculate the distance transform
    distance = ndi.distance_transform_edt(img_array)

    # Generate the watershed markers
    local_maximum = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)))
    markers = ndi.label(local_maximum)[0]

    # Perform watershed and store the labels
    ws_labels = watershed(-distance, markers, mask=img_array)

    # Display the results
    plot_three_images(img_path, img, "Original Image", ms_labels, "MeanShift Labels",
                      ws_labels, "Watershed Labels", output_path_extension + os.path.split(img_path)[1])

    # If you want to visualise the watershed distance markers then try
    # plotting the code below.
    # plot_three_images(img_path, img, "Original Image", -distance, "Watershed Distance",
    #                   ws_labels, "Watershed Labels", output_path_extension + os.path.split(img_path)[1])
