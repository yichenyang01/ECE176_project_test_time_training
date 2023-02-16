import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def add_gaussian_noise(img, mean=0, std=20):
    """
    :param img: an image
    :param mean:
    :param std:
    :return: an image with Gaussian noise added
    """
    # Create Gaussian noise
    noise = np.random.normal(mean, std, img.shape)

    # Apply the noise to the input image
    img_gn = np.zeros(img.shape)
    for i in range(img.shape[2]):
        img_gn[:, :, i] = np.clip(img[:, :, i] + noise[:, :, i], 0, 255)

    return img_gn.astype(np.uint8)


def add_shot_noise(img, lam=100):
    """
    :param img: an image
    :param lam:
    :return: an image with Gaussian noise added
    """
    # Create Poisson noise
    noise = np.random.poisson(lam, img.shape)

    # Apply the noise to the input image
    img_sn = np.zeros(img.shape)
    for i in range(img.shape[2]):
        img_sn[:, :, i] = np.clip(img[:, :, i] + noise[:, :, i], 0, 255)

    return img_sn.astype(np.uint8)


def add_motion_blur(img, kernel_size, direction='h'):
    """
    :param img: an image
    :param kernel_size:
    :param direction: horizontal or vertical
    :return:
    """
    # Create kernel
    kernel = np.zeros((kernel_size, kernel_size))
    if direction == 'h':
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    elif direction == 'v':
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    else:
        raise ValueError('input must be \'h\' or \'v\'')
    kernel /= kernel_size

    # Apply the kernel to the input image
    img_mb = cv2.filter2D(img, -1, kernel)

    return img_mb


def add_defocus_blur(img, radius=3):
    """
    :param img: an image
    :param radius:
    :param blur_amount:
    :return:
    """
    # Convert BGR image to Pillow image
    if not isinstance(img, Image.Image):
        pillow_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pillow_image = img

    # Create filer
    db_filter = ImageFilter.GaussianBlur(radius)

    # Apply the blur filter to the image
    img_db = pillow_image.filter(db_filter)

    return cv2.cvtColor(np.array(img_db), cv2.COLOR_RGB2BGR)


def add_brightness(img, brightness_factor=1.5):
    """
    :param img: an image
    :param brightness_factor:
    :return:
    """
    # Convert BGR image to Pillow image
    if not isinstance(img, Image.Image):
        pillow_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pillow_image = img

    # Enhance brightness to the input image
    img_b = ImageEnhance.Brightness(pillow_image).enhance(brightness_factor)

    return cv2.cvtColor(np.array(img_b), cv2.COLOR_RGB2BGR)

