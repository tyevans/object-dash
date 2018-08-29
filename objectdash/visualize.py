import os
import random
import uuid

import cv2
from django.conf import settings


def get_random_color(num_channels=3):
    x = list(range(0, 255))
    if num_channels == 1:
        return [random.choice(x)]
    if num_channels == 3:
        return [random.choice(x), random.choice(x), random.choice(x)]
    if num_channels == 4:
        return [random.choice(x), random.choice(x), random.choice(x), 255]


def visualize_annotation(image_np, annotations):
    for annotation in annotations:
        annotation.draw(image_np, get_random_color())
    image_id = str(uuid.uuid4())
    image_path = os.path.join("vis/annotations/", image_id + '.jpg')
    image_name = os.path.join(settings.STATIC_ROOT, image_path)
    cv2.imwrite(image_name, image_np)
    return image_path


def crop_annotations(image_np, annotations):
    crops = []
    for annotation in annotations:
        crop_image = annotation.crop(image_np)
        image_id = str(uuid.uuid4())
        image_path = os.path.join("vis/annotations/crops/", image_id + '.jpg')
        image_name = os.path.join(settings.STATIC_ROOT, image_path)
        cv2.imwrite(image_name, crop_image)
        crops.append(image_path)
    return crops
