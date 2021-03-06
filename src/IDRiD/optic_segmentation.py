# Logic to the code is written by sboosi

import cv2
import matplotlib.pyplot as plt
import numpy as np
import src.IDRiD.utils as util


def canny(img, sigma):
    lower = int(max(0, (1.0 - sigma) * np.mean(img)))
    upper = int(min(255, (1.0 + sigma) * np.mean(img)))
    return cv2.Canny(img, lower, upper)


def jaccard_score(input, target, epsilon=1e-6):
    input = input.reshape(-1)
    target = target.reshape(-1)

    # Compute jaccard
    intersect = (input * target).sum()
    union = input.sum() + target.sum()

    return intersect / (union + epsilon - intersect)


def dice_score(input, target, epsilon=1e-6):
    input = input.reshape(-1)
    target = target.reshape(-1)

    # Compute dice
    intersect = (input * target).sum()
    union = input.sum() + target.sum()

    return (2 * intersect) / (union + epsilon)


# Given knowledge
original_image_extension_type = 'jpg'
ground_truth_extension_type = 'tif'

# # Get all the pixels og the original images
original_image_dir = util.get_original_images_dir()
input_images = np.array(util.read_images_from_folder(original_image_dir, original_image_extension_type))

ground_truth_image_dir = util.get_ground_truth_dir()
output_images = np.array(util.read_images_from_folder(ground_truth_image_dir, ground_truth_extension_type))

print(len(input_images))
total_jaccard = 0
total_dice = 0
if input_images.shape[0] == output_images.shape[0]:
    for i in range(input_images.shape[0]):
        # Get a single image
        # Row * Column shape is 2848 * 4288
        input_image = input_images[i]
        output_image = output_images[i]
        input_image_copy = input_image.copy()

        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Add the blurring used before doing the edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        weighted_image = cv2.addWeighted(gray, 1.6, blur, -0.5, 0)
        # Opening morphological transformation
        eroded_image = cv2.erode(weighted_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=2)
        diluted_image = cv2.dilate(eroded_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
        preprocessed_image = cv2.equalizeHist(diluted_image)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(preprocessed_image)

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

        cropped_image = input_image_copy[y0:y1, x0:x1]

        util.save_image('ROI', cropped_image)

        # Apply pre-processing on the retrieved optic disc image
        optic_image = cropped_image.copy()

        # r will remove the blood vessels in the optic disc
        b, g, r = cv2.split(cropped_image)
        blur = cv2.GaussianBlur(r, (5, 5), 5)
        weighted_image = cv2.addWeighted(r, 1.6, blur, -0.5, 0)
        # Closing morphological transformation
        diluted_image = cv2.dilate(weighted_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        eroded_image = cv2.erode(diluted_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=3)

        # Tried Histogram Equalization but has the same effect on the output
        # preprocessed_image = cv2.equalizeHist(eroded_image)
        _, th = cv2.threshold(eroded_image, 250, 255, cv2.THRESH_BINARY)

        # Apply canny edge detection to the threshold output
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
        jaccard_similarity = jaccard_score(prediction, output_image)
        dice_similarity = dice_score(prediction, output_image)

        total_jaccard = total_jaccard + jaccard_similarity
        total_dice = total_dice + dice_similarity
        # save_image('canny', canny_edge)
        # save_image('threshold', th)

average_jaccard = total_jaccard / len(input_images)
average_dice = total_dice / len(input_images)

print('Final Jaccard Sim {}, Final Dice Sim {}'.format(average_jaccard, average_dice))
