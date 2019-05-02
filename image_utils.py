import numpy as np
import cv2
"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""


def random_crop(images, shape=(56, 56, 3)):
    """
    shape 크기만큼 이미지 내에서 무작위로 잘라오는 메소드

    """
    _, h, w, _ = images.shape
    max_h = h - shape[0]
    max_w = w - shape[1]
    crops = []
    for idx, image in enumerate(images):
        start_h = np.random.randint(0, max_h)
        start_w = np.random.randint(0, max_w)
        cropped = image[start_h:start_h + shape[0],
                  start_w:start_w + shape[1]]
        crops.append(cropped)
    return np.stack(crops)


def center_crop(images, shape=(56, 56, 3)):
    """
    shape 크기만큼 이미지 가운데를 오려내는 메소드

    """
    _, h, w, _ = images.shape
    pad_h = h - shape[0]
    pad_w = w - shape[1]

    top_pad = pad_h // 2
    bot_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad

    return images[:, top_pad:-bot_pad, left_pad:-right_pad, :]


def random_flip_left_right(images):
    """
    무작위로 이미지를 좌우로 뒤집어 주는 메소드
    """
    for idx, image in enumerate(images):
        if np.random.random() > 0.5:
            images[idx] = image[:,::-1]
    return images


def random_flip_up_down(images):
    """
    무작위로 이미지를 위 아래로 뒤집어 주는 메소드
    """
    for idx, image in enumerate(images):
        if np.random.random() > 0.5:
            images[idx] = image[::-1]
    return images


def random_hue_correction(images,min_val=0.7,max_val=1.3):
    """
    무작위로 hue 값을 바꾸어주는 메소드
    """
    results = []
    for image in images:
        value = np.random.uniform(min_val,max_val)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:,:,0] = cv2.multiply(image[:,:,0],value)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        results.append(image)
    return np.stack(results)


def random_saturation_correction(images,min_val=0.7,max_val=1.3):
    """
    무작위로 Saturation 값을 바꾸어주는 메소드
    """
    results = []
    for image in images:
        value = np.random.uniform(min_val,max_val)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:,:,1] = cv2.multiply(image[:,:,1],value)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        results.append(image)
    return np.stack(results)


def random_brightness_correction(images,min_val=0.8,max_val=1.2):
    """
    무작위로 Brightness 값을 바꾸어주는 메소드
    """
    results = []
    for image in images:
        value = np.random.uniform(min_val,max_val)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:,:,2] = cv2.multiply(image[:,:,2],value)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        results.append(image)
    return np.stack(results)