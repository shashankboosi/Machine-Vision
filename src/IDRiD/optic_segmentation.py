import cv2
import matplotlib.pyplot as plt

import src.IDRiD.utils as util
import numpy as np
import os

# Given knowledge
original_image_extension_type = 'jpg'
ground_truth_extension_type = 'tif'


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
single_image = np.array(util.read_image(os.path.join(util.get_original_images_dir(), 'IDRiD_17.jpg'), 1))
output_image = np.array(util.read_image(os.path.join(util.get_ground_truth_dir(), 'IDRiD_17_OD.tif'), 0))
image_copy = single_image.copy()
# roi_region = getROI(single_image)
# util.save_image('roi', roi_region)
gray = cv2.cvtColor(single_image, cv2.COLOR_BGR2GRAY)

# Add the blurring used before doing the edge detection
blur = cv2.GaussianBlur(gray, (5, 5), 0)
util.save_image('gaussian', blur)

weighted_image = cv2.addWeighted(gray, 1.6, blur, -0.5, 0)
eroded_image = cv2.erode(weighted_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=2)
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
# util.save_image('gaussian_weighted', gray)
# util.save_image('closing', diluted_image)
# util.save_image('hist', preprocessed_image)
# util.save_image('crop_closing', cropped_image)
# util.save_image('circle', cv2.circle(single_image, maxLoc, 250, (128, 240, 75), 5))


def canny(img, sigma):
    lower = int(max(0, (1.0 - sigma) * np.mean(img)))
    upper = int(min(255, (1.0 + sigma) * np.mean(img)))
    return cv2.Canny(img, lower, upper)


def jaccard_score(input, target, epsilon=1e-6):
    # To avoid zero in the numerator
    # smooth = 1
    input = input.reshape(-1)
    # print('Input dice', input.size())
    target = target.reshape(-1)
    # print('Target dice', target.size())

    # Compute dice
    intersect = (input * target).sum()
    union = input.sum() + target.sum()

    return intersect / (union + epsilon - intersect)


def dice_score(input, target, epsilon=1e-6):
    # To avoid zero in the numerator
    # smooth = 1
    input = input.reshape(-1)
    # print('Input dice', input.size())
    target = target.reshape(-1)
    # print('Target dice', target.size())

    # Compute dice
    intersect = (input * target).sum()
    union = input.sum() + target.sum()

    return (2 * intersect) / (union + epsilon)


# Apply pre-processing on the retrieved optic disc image
optic_image = cropped_image.copy()
print(optic_image.shape)
# r will remove the blood vessels in the optic disc
b, g, r = cv2.split(cropped_image)
blur = cv2.GaussianBlur(r, (5, 5), 5)
weighted_image = cv2.addWeighted(r, 1.6, blur, -0.5, 0)
diluted_image = cv2.dilate(weighted_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
eroded_image = cv2.erode(diluted_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=3)
# th = cv2.adaptiveThreshold(eroded_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# preprocessed_image = cv2.equalizeHist(eroded_image)
ret_val, th = cv2.threshold(eroded_image, 250, 255, cv2.THRESH_BINARY)

canny_edge = canny(th, 0.20)

# Image display of the prediction
prediction_canny = np.zeros((2848, 4288))
prediction_threshold = np.zeros((2848, 4288))
prediction_canny[y0:y1, x0:x1] = canny_edge
prediction_threshold[y0:y1, x0:x1] = th
prediction_threshold[prediction_threshold == 255] = 127
util.save_image('prediction_canny', prediction_canny)
util.save_image('prediction_threshold', prediction_threshold)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(output_image, cmap='gray')
ax1.set_title('Input Image')
ax1.axis('off')

ax2.imshow(prediction_threshold, cmap='gray')
ax2.set_title('Predicted Image')
ax2.axis('off')
plt.show()

# Calculate jaccard and dice score
prediction = np.zeros((2848, 4288))
prediction[y0:y1, x0:x1] = th
prediction[prediction == 255] = 1
output_image[output_image == 76] = 1
print(jaccard_score(prediction, output_image))
print(dice_score(prediction, output_image))

# util.save_image('gaussian', blur)
# util.save_image('r', r)
# util.save_image('gaussian_weighted_crop', weighted_image)
# util.save_image('closing_crop', eroded_image)
# util.save_image('hist_Crop', preprocessed_image)
# util.save_image('canny', canny_edge)
# util.save_image('threshold', th)
