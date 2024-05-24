import base64
import hashlib
import json
import os
from io import BytesIO

from PIL import Image


def count_images_in_dir(directory):
    types = ('.jpg', '.png', '.jpeg')  # the tuple of file types
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(types):
                count += 1
    return count


def dict_to_hash(d):
    d_json = json.dumps(d, sort_keys=True)
    return hashlib.sha256(d_json.encode('utf-8')).hexdigest()


def img_2_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_2_img(base64_string: str) -> Image.Image:
    binary_data = base64.b64decode(base64_string)
    image_bytes = BytesIO(binary_data)
    pillow_image = Image.open(image_bytes)
    return pillow_image


def read_base64_image(path: str, as_image=False) -> str | Image.Image:
    with open(path, "r") as f:
        base64_string = f.read().strip()
    if as_image:
        return base64_2_img(base64_string)
    return base64_string


def resize_image(image: Image.Image, new_size: int) -> Image.Image:
    original_size = min(image.size)
    if original_size <= new_size:
        return image
    k = new_size / original_size
    return image.resize((int(image.width * k), int(image.height * k)), Image.LANCZOS)


def convert_number_to_string(number: int) -> str:
    if 0 <= number <= 99999999:
        return f"{number:08d}"
    raise ValueError("number not in interval 0 - 99999999")


def prepare_number(key: str | int):
    if isinstance(key, str):
        if len(key) == 8:
            return key
        raise ValueError("key must be of length 8")
    return convert_number_to_string(key)


def square_bbox(bbox):
    """
    Make the bounding box square by extending the shorter side to match the longer side.
    :param bbox: [x1, y1, x2, y2] - upper left and bottom right corners of the bounding box
    :return: bbox: [x1, y1, x2, y2] - upper left and bottom right corners of the bounding box, the box is a square
    """
    bbox_new = bbox.copy()
    mid_x = (bbox[0] + bbox[2]) // 2
    mid_y = (bbox[1] + bbox[3]) // 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    m = max(width, height)
    bbox_new[0] = mid_x - m // 2
    bbox_new[2] = mid_x + m // 2
    bbox_new[1] = mid_y - m // 2
    bbox_new[3] = mid_y + m // 2
    return bbox_new
