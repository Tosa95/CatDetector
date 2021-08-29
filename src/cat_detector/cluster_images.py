import os
import pickle
import time
import traceback
from pprint import pprint
from random import shuffle

import cv2
import numpy as np
from cat_detector.bisecting_kmeans import nsecting_kmeans
from matplotlib import pyplot as plt
from paths import IMAGES_FOLDER, CLUSTERING_FILE

HUE_MAX = 179
HUE_HR_MAX = 360
SATURATION_MAX = 255
VALUE_MAX = 255

NORMALIZED_MAX = 255

DISPLAY_IMAGES = True


def image_generator(img_folder: str, name_contains: str = "_curr.jpg"):
    all_files = list(os.listdir(img_folder))
    shuffle(all_files)

    for img_file_name in all_files:
        if name_contains in img_file_name:
            img_path = os.path.join(img_folder, img_file_name)

            img = cv2.imread(img_path)

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            yield img_file_name, img, img_hsv


def rescale(I: np.ndarray, curr_max: float, new_max: float):
    return I.astype("float32") / curr_max * new_max


def hue_diff(target_hue: float, hue: np.ndarray):
    abs_diff = np.abs(target_hue - hue)
    abs_diff_inverted = 360.0 - abs_diff
    return np.minimum(abs_diff, abs_diff_inverted)


def segment_background_image(img_hsv: np.ndarray, target_hue: float,
                             target_saturation: float, diff_tolerance: float):
    segmented_img = np.copy(img_hsv).astype("float32")

    H = rescale(img_hsv[:, :, 0], HUE_MAX, HUE_HR_MAX)
    S = rescale(img_hsv[:, :, 1], SATURATION_MAX, 1.0)

    H_diff = hue_diff(target_hue, H)
    H_diff[S < target_saturation] = 360.0

    segmented_img[H_diff < diff_tolerance] = -1

    return segmented_img


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


if __name__ == "__main__":
    infos = []

    for i, (img_name, img, img_hsv) in enumerate(image_generator(IMAGES_FOLDER)):

        print(f"--> {i}")

        segmented_img = segment_background_image(img_hsv, 200, 0.4, 30)

        segmented_img[:, :, 0] = rescale(segmented_img[:, :, 0], HUE_MAX, 1)
        segmented_img[:, :, 1] = rescale(segmented_img[:, :, 1], SATURATION_MAX, 1)
        segmented_img[:, :, 2] = rescale(segmented_img[:, :, 2], VALUE_MAX, 1)

        points = np.reshape(segmented_img, (segmented_img.shape[0] * segmented_img.shape[1], -1))
        total_points = points.shape[0]
        points = points[points[:, 0] > 0]

        total_segmented = np.count_nonzero(points[:, 0] > 0)
        segmentation_ratio = total_segmented / total_points

        print(f"Segmented points: {total_segmented}, Segmentation ratio: {segmentation_ratio}")

        positions = np.random.choice(range(points.shape[0]), size=1000, replace=False)

        selected_points = points[positions, :]

        try:
            before = time.time()
            labels, info = nsecting_kmeans(selected_points,
                                           lambda clusters: sorted(clusters, key=lambda x: x.avg_diff_from_mean)[
                                               -1].label,
                                           lambda clusters: (max([c.avg_diff_from_mean for c in clusters]) < 0.1)
                                                            or len(clusters) >= 30, max_iter=20, final_kmeans=False)
            print(time.time() - before)

            info = sorted(info, key=lambda x: x.size, reverse=True)

            infos.append((img_name, info, segmentation_ratio))

            pprint(info)
            print(len(info))

            if DISPLAY_IMAGES:

                segmentedRGB = np.copy(img)
                segmentedRGB[segmented_img < 0] = 0

                cv2.imshow("orig", img)
                cv2.imshow("segmented", segmentedRGB)
                k = cv2.waitKey(0)
                if k == 103:
                    fig = plt.figure()
                    plt.style.use('dark_background')
                    ax = fig.add_subplot(projection='3d')
                    ax.scatter(rand_jitter(selected_points[:, 0]), rand_jitter(selected_points[:, 1]),
                               rand_jitter(selected_points[:, 2]), s=1, c=labels, cmap="hsv")
                    ax.set_xlabel('Hue')
                    ax.set_ylabel('Saturation')
                    ax.set_zlabel('Value')
                    plt.show()

        except Exception as e:
            traceback.print_exc()

        if i % 5000 == 0 and i != 0:
            with open(os.path.join(CLUSTERING_FILE), "wb") as f:
                pickle.dump(infos, f)

    with open(os.path.join(CLUSTERING_FILE), "wb") as f:
        pickle.dump(infos, f)
