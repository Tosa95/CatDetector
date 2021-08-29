import json
import os
from copy import deepcopy

import cv2
import numpy as np
import tensorflow as tf
from cat_detector.config import CLUSTERS_TO_KEEP
from cat_detector.nn_training import prepare_clusterings_for_model
from cat_detector.paths import IMAGES_FOLDER, LABELS_FILE, load_clusterings, load_labels, MODEL_FILE
from matplotlib import pyplot as plt


def save_labels(labels):
    with open(LABELS_FILE, "wt") as f:
        json.dump(labels, f)


def show_image(img_name):
    img_path = os.path.join(IMAGES_FOLDER, img_name)
    img = cv2.imread(img_path)
    cv2.imshow("orig", img)


def wait_for_key():
    return chr(cv2.waitKey(0))


def get_label_distr(y, labels, labels_map):
    all_labels = np.array([labels_map[l] for _, l in labels.items() if l in labels_map])
    n_labels = y.shape[-1]
    return np.array([np.count_nonzero(all_labels == l) for l in range(n_labels)])


def score_images_less_represented_first(y, labels, labels_map):
    n = get_label_distr(y, labels, labels_map)
    category_scores = 1.0 / n
    return np.sum(category_scores * y, axis=1).tolist()


def score_images_more_uncertain_first(y):
    scores = -np.sum(y * np.log2(y), axis=1)
    scores[np.isnan(scores)] = 0.0

    return scores


def combine_scores(exponents, *scores):
    scores = np.transpose(scores)
    scores = np.power(scores, exponents)
    return np.prod(scores, axis=1)


def print_stats(y, labels, labels_map, scores):
    labels_predicted = np.argmax(y, axis=1)

    covid = labels_predicted[labels_predicted == 0].shape[0]
    kiko = labels_predicted[labels_predicted == 1].shape[0]

    print(f"Dataset distr: covid: {covid / (covid + kiko)}, kiko: {kiko / (covid + kiko)}")
    print(f"Labels distr: {get_label_distr(y, labels, labels_map)}")

    plt.plot(np.sort(scores)[::-1])
    plt.show()


def do_labeling(model):
    labels_map = {"c": 0, "k": 1, "n": 2}

    labels = {}

    if os.path.exists(LABELS_FILE):
        labels = load_labels()

    clustering = [(img_name, clustering_info, segmentation_ratio) for img_name, clustering_info, segmentation_ratio in
                  load_clusterings() if img_name not in labels]

    x = prepare_clusterings_for_model(clustering, clusters_to_keep=CLUSTERS_TO_KEEP)

    y = model.predict(x)

    scores_less_repr = score_images_less_represented_first(y, labels, labels_map)
    scores_more_uncertain = score_images_more_uncertain_first(y)

    scores = combine_scores([1, 0.2], scores_less_repr, scores_more_uncertain)

    labels_predicted = np.argmax(y, axis=1)

    clustering = [(scores[i], labels_predicted[i], y[i], x[i], img_name, clustering_info) for
                  i, (img_name, clustering_info, segmentation_ratio) in enumerate(clustering)]

    clustering = sorted(clustering, key=lambda x: x[0], reverse=True)

    print_stats(y, labels, labels_map, scores)

    n = 0
    for i, (score, label_predicted, y, x, img_name, clustering_info) in enumerate(clustering):

        if img_name not in labels:
            show_image(img_name)
            print(score)
            print(label_predicted)
            print(y)
            print(x)
            labels[img_name] = wait_for_key()

            n += 1

            if n % 10 == 0:
                save_labels(labels)

    save_labels(labels)


def correct_labeling():
    labels = load_labels()

    labels_res = deepcopy(labels)

    n = 0
    for img_name, label in labels.items():
        show_image(img_name)
        print(label)
        resp = wait_for_key()

        if resp != " ":
            labels_res[img_name] = resp

        n += 1

        if n % 10 == 0:
            save_labels(labels_res)


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_FILE)
    model.summary()

    do_labeling(model)
    # correct_labeling()
