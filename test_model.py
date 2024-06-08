import os

import cv2

from inference import AgeSexInference
from utils.imgs import read_rgb_image


def main():
    model = AgeSexInference("checkpoints/model_inception_resnet.onnx")
    path = os.path.join("data/test_faces/imgs")
    for img_name in os.listdir(path):
        img = read_rgb_image(os.path.join(path, img_name))
        img_img = cv2.imread(os.path.join(path, img_name))
        img = cv2.resize(img, (160, 160))
        age_label, sex_label = model.predict_labels(img)
        age, sex = model.predict_probs(img)
        print(age, sex, age_label, sex_label)
        # cv2.putText(img_img, f"Age: {age[0]}, Sex: {sex[0]}", (1, 200),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2, cv2.LINE_AA)
        cv2.imshow("img", img_img)
        cv2.waitKey(0)

    # for i in random.sample(list(range(100, 1800, 3)), 300):
    #     img = read_rgb_image(path)
    #     img_img = cv2.imread(path)
    #     age, sex = model.predict_labels(img)
    #     cv2.putText(img_img, f"Age: {age[0]}, Sex: {sex[0]}", (1, 200),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2, cv2.LINE_AA)
    #     cv2.imshow('img', img_img)
    #     cv2.waitKey(0)


if __name__ == "__main__":
    main()
