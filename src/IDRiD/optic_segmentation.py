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
    width = int(img.shape[1] / 4)
    height = int(img.shape[0] / 4)
    #####
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


# Get a single image
# Row * Column shape is 2848 * 4288
single_image = np.array(util.read_image(os.path.join(util.get_original_images_dir(), 'IDRiD_10.jpg'), 1))
image_copy = single_image.copy()
# roi_region = getROI(single_image)
# util.save_image('roi', roi_region)
gray = cv2.cvtColor(single_image, cv2.COLOR_BGR2GRAY)

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

weighted_image = cv2.addWeighted(gray, 1.6, blur, -0.5, 0)
eroded_image = cv2.erode(weighted_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=3)
diluted_image = cv2.dilate(eroded_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
preprocessed_image = cv2.equalizeHist(diluted_image)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(preprocessed_image)
print(maxVal, maxLoc)

# Cut the image to get the optic disk portion of it
shift_equal = 510
if int(maxLoc[0]) - shift_equal >= 0:
    x0 = int(maxLoc[0]) - shift_equal
else:
    x0 = 0
if int(maxLoc[1]) - shift_equal >= 0:
    y0 = int(maxLoc[1]) - shift_equal
else:
    y0 = 0
if int(maxLoc[0]) + shift_equal <= 4288:
    x1 = int(maxLoc[0]) + shift_equal
else:
    x1 = 4288
if int(maxLoc[1]) + shift_equal <= 2848:
    y1 = int(maxLoc[1]) + shift_equal
else:
    y1 = 2848

cropped_image = image_copy[y0:y1, x0:x1]
util.save_image('gaussian_weighted', gray)
util.save_image('closing', diluted_image)
util.save_image('hist', preprocessed_image)
util.save_image('crop_closing', cropped_image)
util.save_image('circle', cv2.circle(single_image, maxLoc, 250, (128, 240, 75), 5))

# Apply pre-processing on the retrieved optic disc image
optic_image = cropped_image
print(optic_image.shape)
# r will remove the blood vessels in the optic disc
b, g, r = cv2.split(optic_image)
util.save_image('r', r)



# home = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
# mask1 = cv2.Canny(home, 100, 300)
# util.save_image('canny1', mask1)
# mask1 = cv2.GaussianBlur(mask1, (1, 1), 0)
# mask1 = cv2.Canny(mask1, 100, 300)
# util.save_image('morph', home)
# util.save_image('canny2', mask1)

# display the results of the naive attempt
util.save_image('Naive', single_image)

print(util.save_image('resize', util.resize(single_image)))
# print(single_image.transpose((1, 0, 2))[0][200:500])



# for i in xrange(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
#
# plt.show()
