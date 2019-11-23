import cv2
import os
import glob

def read_image(image_name, type):
    return cv2.imread(image_name, type)


def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(file_name, image):
    cv2.imwrite(file_name + ".jpg", image)


# Get the directory containing of the train and test dataset dynamically
def get_dir_path_with_train_test_dirs():
    project_dir = os.path.abspath('../..')
    if 'Images' in os.listdir(project_dir):
        return os.path.join(project_dir, 'Images', 'IDRiD')
    else:
        exit('Wrong check to the directory, please check your code!')


# Get the original images folder
def get_original_images_dir():
    if 'original_retinal_images' in os.listdir(get_dir_path_with_train_test_dirs()):
        return os.path.join(get_dir_path_with_train_test_dirs(), 'original_retinal_images')
    else:
        exit('Check if the dataset is removed from the dir')


# Get the ground truth folder
def get_ground_truth_dir():
    if 'optic_disc_segmentation_masks' in os.listdir(get_dir_path_with_train_test_dirs()):
        return os.path.join(get_dir_path_with_train_test_dirs(), 'optic_disc_segmentation_masks')
    else:
        exit('Check if the dataset is removed from dir')


def read_images_from_folder(dir_path, image_extension_type):
    image_list = []
    image_path = os.path.join(dir_path, '*.' + image_extension_type)
    for image_file_name in glob.glob(image_path):
        if image_extension_type == 'jpg':
            image = read_image(image_file_name, 1)
        else:
            image = read_image(image_file_name, 0)
        if image is not None:
            image_list.append(image)

    return image_list
