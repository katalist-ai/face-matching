import cv2
from PIL import Image
import numpy as np
from time import perf_counter
from progress_manager import prepare_number

def get_image_path(number):
    return f"data/images/{number}.png"

def test_cv2(image_path):
    start_time = perf_counter()
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    end_time = perf_counter()
    return end_time - start_time

def test_cv2_2(image_path):
    start_time = perf_counter()
    bgr_image = cv2.imread(image_path)[..., ::-1]
    end_time = perf_counter()
    return end_time - start_time

def test_cv2_3(image_path):
    start_time = perf_counter()
    bgr_image = cv2.imread(image_path)[..., [2, 1, 0]]
    end_time = perf_counter()
    return end_time - start_time

def test_pil(image_path):
    start_time = perf_counter()
    pil_image = Image.open(image_path)
    numpy_image = np.asarray(pil_image)
    end_time = perf_counter()
    return end_time - start_time

if __name__ == "__main__":
    times = [0,0,0,0]
    fns = [test_cv2, test_cv2_2, test_cv2_3, test_pil]
    N = 100
    for n, f in enumerate(fns):
        for i in range(N):
            # Test OpenCV
            k = prepare_number(i)
            img_path = get_image_path(k)
            time = f(img_path)
            times[n] += time
        times[n] /= N
    print(times)