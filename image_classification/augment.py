"""
See https://github.com/imjeffhi4/pokemon-classifier/blob/main/data_collection/augment_data.ipynb
"""

import os
import random
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm


def gaussian_blur(image: Image, radius_range: Tuple[int, int] = (1, 5)) -> Image:
    """
    Adds a Gaussian Blur effect to the image using a randomly picked Standard deviation for the Gaussian kernel.

    :param image: image to modify
    :param radius_range: tuple of (lower limit, upper limit) of the Gaussian kernel Standard deviation to pick from
    :return: modified image
    """
    radius = random.randint(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def rotate(image: Image, max_deg: int = 30) -> Image:
    """
    Adds a randomly picked rotation effect of up to max_deg degrees clockwise/counterclockwise.

    :param image: image to modify
    :param max_deg: maximum degree to rotate the image by
    :return: modified image
    """
    degree = random.choice(list(range(0, max_deg)) + list(range(360, 360 - max_deg)))
    return image.rotate(degree)


def mirror(image: Image) -> Image:
    """
    Mirrors the image horizontally.

    :param image: image to modify
    :return: modified image
    """
    return ImageOps.mirror(image)


def quantizing(image):
    """
    Effectively reduces the number of colors in an image by the shift_amount
    See https://github.com/imjeffhi4/pokemon-classifier/blob/main/data_collection/augment_data.ipynb
    """
    shift_amount = random.randint(4, 7)
    red = (np.asarray(image)[:, :, 0] >> shift_amount) << shift_amount
    green = (np.asarray(image)[:, :, 1] >> shift_amount) << shift_amount
    blue = (np.asarray(image)[:, :, 2] >> shift_amount) << shift_amount
    return Image.fromarray(np.stack((red, green, blue), axis=2))


def add_noise(image):
    """
    Adds Gaussian noise to the image
    See https://github.com/imjeffhi4/pokemon-classifier/blob/main/data_collection/augment_data.ipynb
    """
    rand_decimal = random.randint(20, 70) / 100  # number between 0.2 and 0.7
    # mean = 0, standard deviation = rand_decimal
    image = np.array(image)
    gaussian = np.random.normal(0, rand_decimal, image.size)
    gaussian = gaussian.reshape(
        image.shape[0], image.shape[1], image.shape[2]).astype('uint8')  # reshaping
    return Image.fromarray(cv2.add(image, gaussian))  # Adding gaussian noise to image


def crop_image(image, left=(5, 25), right=(5, 25), top=(5, 25), bottom=(5, 25)):
    """
    Crops the image and then returns back to original size
    See https://github.com/imjeffhi4/pokemon-classifier/blob/main/data_collection/augment_data.ipynb
    """
    base_size = 224
    left = random.randint(left[0], left[1])
    right = random.randint(base_size - right[1], base_size - right[0])
    top = random.randint(top[0], top[1])
    bottom = random.randint(base_size - bottom[1], base_size - bottom[0])
    return image.crop((left, top, right, bottom)).resize((base_size, base_size))


def add_random_augmention(image: Image) -> Image:
    """
    Adds a randomly selected augmentation effect to the image.

    :param image: image to augment
    :return: augmented image
    """
    random_int = random.randint(1, 6)
    if random_int == 1:
        return gaussian_blur(image)
    elif random_int == 2:
        return rotate(image)
    elif random_int == 3:
        return mirror(image)
    elif random_int == 4:
        return quantizing(image)
    elif random_int == 5:
        return add_noise(image)
    else:
        return crop_image(image)


def create_augmented_dataset(original_dir: str, augmented_dir: str, aug_number: int, test_ratio: float, seed: int):
    """
    Creates a balanced version of an image dataset and adds augmented images to enlarge the dataset.

    :param original_dir: relative path to the unaugmented dataset (folder containing subfolders where
                         each subfolder is one label)
    :param augmented_dir: relative path to the augmented dataset folder to create
    :param aug_number: number of augmented images to add to each original image for the train set
    :param test_ratio: relative size of test_set (between 0 and 1)
    :param seed: seed to use for sampling
    """
    random.seed(seed)

    # balance image count per label:
    # get label with smallest number of images, then sample this number of file names from each labels image dir
    labels = os.listdir(original_dir)
    min_count = min([len(os.listdir(original_dir + label + "/")) for label in labels])
    print("Balanced size per label:", min_count)
    file_dict = {
        label: random.sample(os.listdir(original_dir + label + "/"), min_count)
        for label in labels
    }
    test_size = int(min_count * test_ratio)

    for label, file_list in file_dict.items():
        os.makedirs(augmented_dir + "train/" + label + "/", exist_ok=True)
        os.makedirs(augmented_dir + "test/" + label + "/", exist_ok=True)
        for ind, img_name in enumerate(tqdm(file_list, desc="Running augmentation for label " + label)):
            if ind < test_size:
                set_type = "test"
            else:
                set_type = "train"
            img = Image.open(original_dir + label + "/" + img_name).resize((224, 224))
            # always save original image in augmented folder as well
            img.save(augmented_dir + set_type + "/" + label + "/" + img_name[:-5] + "_" + "0" + ".png")
            # generate aug_number augmented images: always at least one augmentation, 20% chance for a second one.
            # only augment for train set, test set only contains original images
            if set_type == "train":
                for aug_index in range(1, aug_number + 1):
                    img_aug = add_random_augmention(img)
                    if random.random() > 0.8:
                        img_aug = add_random_augmention(img_aug)
                    img_aug.save(augmented_dir + set_type + "/" + label + "/" + img_name[:-5] + "_" + str(
                        aug_index) + ".png")


if __name__ == '__main__':
    create_augmented_dataset(original_dir="/home/mcc/PycharmProjects/Transformer-Silhouettes_v1/image_classification/Dataset/", augmented_dir="/home/mcc/PycharmProjects/Transformer-Silhouettes_v1/image_classification/Dataset_Augmented", aug_number=5, test_ratio=0.2, seed=1)
