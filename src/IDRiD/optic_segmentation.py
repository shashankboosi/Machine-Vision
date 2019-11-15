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

# Get a single image
single_image = np.array(util.read_image(os.path.join(util.get_original_images_dir(), 'IDRiD_01.jpg'), 1))
im = single_image.copy()
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

gray = cv2.addWeighted(gray, 0.8, blur, 0.1, 0)
util.save_image('gaussian_weighted', gray)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
print(maxVal, maxLoc)
util.save_image('circle', cv2.circle(single_image, maxLoc, 5, (255, 0, 0), 2))

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
