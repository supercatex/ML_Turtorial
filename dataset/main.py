import os
import cv2
import numpy as np


_INPUT_DIR = "./images/"
_OUTPUT_DIR = "./output/"
_NUM_OF_SAMPLES = 1000
_SAMPLE_SIZE = (100, 100)


def add_noise(img):
    h, w, c = img.shape
    if c != 4:
        raise Exception("Only PNG format supported!")

    dst = img.copy()

    tmp = dst[:, :, 0:3]
    hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
    p1 = np.random.randint(255 - 35, 255 + 35) / 255
    hsv[:, :, 1] = np.array(hsv[:, :, 1] * p1, dtype=np.uint8)
    p2 = np.random.randint(255 - 100, 255) / 255
    hsv[:, :, 2] = np.array(hsv[:, :, 2] * p2, dtype=np.uint8)
    tmp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    dst[:, :, 0:3] = tmp

    for _ in range(int(h * w * 0.1)):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        dst[y, x] = np.random.randint(0, 255, 4)
        dst[y, x, 3] = 255
    return dst


def generate_image(img):
    h, w, c = img.shape
    if c != 4:
        raise Exception("Only PNG format supported!")

    pts1 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])
    pts2 = np.float32([
        [np.random.randint(0, h / 4), np.random.randint(0, w / 4)],
        [h - np.random.randint(0, h / 4), np.random.randint(0, w / 4)],
        [np.random.randint(0, h / 4), w - np.random.randint(0, w / 4)],
        [h - np.random.randint(0, h / 4), w - np.random.randint(0, w / 4)]
    ])

    m = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, m, (w, h))
    tmp = dst.copy()

    r = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
    r[:, :, c - 1] = 255
    for i in range(c - 1):
        tmp[:, :, i] = cv2.bitwise_and(r[:, :, i], 255 - dst[:, :, c - 1])
        dst[:, :, i] = cv2.bitwise_and(dst[:, :, i], dst[:, :, c - 1])
    dst += tmp

    return dst


def run():
    global _INPUT_DIR, _OUTPUT_DIR, _NUM_OF_SAMPLES, _SAMPLE_SIZE

    for f in os.listdir(_INPUT_DIR):
        print("Processing:", f)

        label_dir = os.path.join(_OUTPUT_DIR, f.split(".")[0][1:].zfill(3))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        image_filename = os.path.join(_INPUT_DIR, f)
        image = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)

        for i in range(_NUM_OF_SAMPLES):
            img = add_noise(image)
            img = generate_image(img)
            img = cv2.resize(img, _SAMPLE_SIZE)
            img_filename = os.path.join(label_dir, "%d.jpg" % i)
            cv2.imwrite(img_filename, img)


if __name__ == "__main__":
    run()
