import cv2
import matplotlib.pyplot as plt
import src.IDRiD.utils as util
import numpy as np
import os
import glob
import scipy.ndimage as ndimage

# Given knowledge
original_image_extension_type = 'jpg'
ground_truth_extension_type = 'tif'

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(len(flags))


def get_command_line_args():
    parser = argparse.ArgumentParser(description='Get info about the task',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_category', '-m', default='Train',
                        help="Specify the model category like train / test")
    parser.add_argument('--directory', '-dir', metavar='INPUT',
                        help='Directory name of images', required=True)
    parser.add_argument('--image_format', '-imf', metavar='INPUT',
                        help='Image formatting type', required=True, default="jpg")
    parser.add_argument('--learning_rate', '-lr', type=float,
                        help="Learning rate of the model",
                        default=0.001)
    parser.add_argument('--batch_size', '-b', type=int,
                        help="Batch size of the model",
                        default=8)
    parser.add_argument('--no_epochs', '-noe', type=int,
                        help="No of epochs for the model",
                        default=5)
    parser.add_argument('--val_size', '-vs', type=int,
                        help="Validation Size",
                        default=0.2)
    parser.add_argument('--blur_type', '-blur', help="Enter the type of blur ( median/ gaussian)",
                        default="gaussian")

    return parser.parse_args()


# # Get all the pixels og the original images
# original_image_dir = util.get_original_images_dir()
# image_pixels = np.array(util.read_images_from_folder(original_image_dir, original_image_extension_type))
#
# print(image_pixels.shape)
def resize(img):
    width = 1024
    height = 720
    #####
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def getROI(image):
    image_resized = resize(image)
    b, g, r = cv2.split(image_resized)
    g = cv2.GaussianBlur(g, (15, 15), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    g = ndimage.grey_opening(g, structure=kernel)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(g)

    x0 = int(maxLoc[0]) - 110
    y0 = int(maxLoc[1]) - 110
    x1 = int(maxLoc[0]) + 110
    y1 = int(maxLoc[1]) + 110

    return image_resized[y0:y1, x0:x1]


# Get a single image
single_image = np.array(util.read_image(os.path.join(util.get_original_images_dir(), 'IDRiD_03.jpg'), 1))
im = single_image.copy()
roi_region = getROI(single_image)
util.save_image('roi', roi_region)
print(single_image.shape)
gray = cv2.cvtColor(single_image, cv2.COLOR_BGR2GRAY)
util.save_image('gray', gray)

# Add the blurring used before doing the edge detection
type = "gaussian"
blur = []
if type is "median":
    blur = cv2.medianBlur(gray, 5, 0)
    util.save_image('median', blur)
elif type is "gaussian":
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    util.save_image('gaussian', blur)
else:
    exit("Please check the blur types we support for the optic segmentation")

gray = cv2.addWeighted(gray, 1.6, blur, -0.5, 0)
img = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=3)
img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=2)
util.save_image('closing', img)
img = cv2.equalizeHist(img)
util.save_image('hist', img)
# home = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
# mask1 = cv2.Canny(home, 100, 300)
# util.save_image('canny1', mask1)
# mask1 = cv2.GaussianBlur(mask1, (1, 1), 0)
# mask1 = cv2.Canny(mask1, 100, 300)
# util.save_image('morph', home)
# util.save_image('canny2', mask1)

util.save_image('gaussian_weighted', gray)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
print(maxVal, maxLoc)
util.save_image('circle', cv2.circle(single_image, maxLoc, 250, (128, 240, 75), 5))

# display the results of the naive attempt
util.save_image('Naive', single_image)

print(util.save_image('resize', util.resize(single_image)))
# print(single_image.transpose((1, 0, 2))[0][200:500])

# Row * Column shape is 2848 * 4288
b, g, r = cv2.split(single_image)
util.save_image('g', g)
util.save_image('r', r)
util.save_image('b', b)

# for i in xrange(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()
