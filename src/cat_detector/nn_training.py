import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from cat_detector.bisecting_kmeans import ClusterInfo
from cat_detector.config import CLUSTERS_TO_KEEP
from cat_detector.paths import load_clusterings, load_labels, MODEL_FILE, MODEL_FILE_NO_SOFTMAX
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


def prepare_clusterings_for_model(clusterings, clusters_to_keep):
    resized = make_all_clusters_same_length(
        [(img_name, "", clstr, clustering_ratio) for img_name, clstr, clustering_ratio in clusterings],
        clusters_to_keep)
    resized = [(img_name, clstr, segmentation_ratio) for img_name, _, clstr, segmentation_ratio in resized]

    x = np.array(
        [[c.centroid.tolist() + [c.size / 1000, c.avg_diff_from_mean, segmentation_ratio] for c in
          clustering[:clusters_to_keep]] for
         _, clustering, segmentation_ratio
         in resized])

    x[np.isnan(x)] = 0.0

    return x.reshape(x.shape[0], -1)


def make_all_clusters_same_length(labels_and_clustering: List[Tuple[str, str, List[ClusterInfo], float]],
                                  length: int) -> List[
    Tuple[str, str, List[ClusterInfo], float]]:
    res = []

    for img_name, lbl, clusters, segmentation_ratio in labels_and_clustering:
        clusters = clusters[:length]
        if len(clusters) < length:
            clusters.extend([ClusterInfo(-1, np.zeros(clusters[0].centroid.shape), 0, 0)] * (length - len(clusters)))
        res.append((img_name, lbl, clusters, segmentation_ratio))

    return res


def extract_labeled_clusterings():
    clusterings = load_clusterings()
    labels = load_labels()

    return [(img_name, labels[img_name], clustering_info, segmentation_ratio)
            for img_name, clustering_info, segmentation_ratio in clusterings if img_name in labels]


def group_by_labels(labels_and_clustering):
    lbls = [l for _, l, _, _ in labels_and_clustering]
    cntr = Counter(lbls)

    res = {}

    for key in cntr.keys():
        res[key] = [(img_name, clustering_info, segmentation_ratio) for img_name, l, clustering_info, segmentation_ratio
                    in labels_and_clustering if l == key]

    return res


def train(x, y, n_labels, hidden_size, epochs):
    y = to_categorical(y, n_labels)

    train_split = int(0.6 * x.shape[0])
    test_split = int(0.2 * x.shape[0] + train_split)
    x_train, x_test, x_validate = np.split(x, [train_split, test_split])
    y_train, y_test, y_validate = np.split(y, [train_split, test_split])

    hidden1 = keras.layers.Dense(hidden_size, activation='relu')
    hidden2 = keras.layers.Dense(hidden_size, activation='relu')
    final = keras.layers.Dense(3)

    input = keras.layers.Input(shape=(x.shape[1],))
    l = hidden1(input)
    l = keras.layers.Dropout(0.5)(l)
    l = hidden2(l)
    l = keras.layers.Dropout(0.5)(l)
    output_no_softmax = final(l)
    output = keras.layers.Softmax()(output_no_softmax)

    model = keras.Model(input, output)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x, y, epochs=epochs, batch_size=64, validation_data=(x_validate, y_validate))

    model.save(MODEL_FILE)

    l2 = hidden1(input)
    l2 = hidden2(l2)
    output_no_softmax = final(l2)

    model_no_softmax = keras.Model(input, output_no_softmax)

    model_no_softmax.save(MODEL_FILE_NO_SOFTMAX)

    return np.percentile(history.history["val_loss"], 1), np.percentile(history.history["val_accuracy"], 99)


if __name__ == "__main__":

    clusters_to_keep = CLUSTERS_TO_KEEP

    labels_and_clusterings = extract_labeled_clusterings()
    labels_and_clusterings = make_all_clusters_same_length(labels_and_clusterings, clusters_to_keep)

    grouped_by_labels = {k: v for k, v in group_by_labels(labels_and_clusterings).items() if k in ["c", "k", "n"]}

    representations = {}

    # samples_per_group = max(len(entries) for entries in grouped_by_labels.values())
    samples_per_group = min(len(entries) for entries in grouped_by_labels.values())

    labels = sorted(grouped_by_labels.keys())

    x = []
    y = []

    print(samples_per_group)

    for i, l in enumerate(["c", "k", "n"]):
        features = [[c.centroid.tolist() + [c.size / 1000, c.avg_diff_from_mean, segmentation_ratio] for c in
                     clustering[:clusters_to_keep]]
                    for _, clustering, segmentation_ratio
                    in grouped_by_labels[l] if len(clustering) >= clusters_to_keep]
        random.shuffle(features)
        features = features[:samples_per_group]
        x.extend(features)
        y.extend([i] * len(features))

    x = np.array(x)
    x = x.reshape(x.shape[0], -1)
    x[np.isnan(x)] = 0.0
    y = np.array(y)

    p = np.random.permutation(x.shape[0])
    x = x[p, :]
    y = y[p]

    print(x.shape)
    print(y.shape)

    best_losses = []
    tf.get_logger().setLevel('INFO')

    # for hs in range(8, 129, 8):
    #     fixed_hs_losses = np.array([train(np.array(x), np.array(y), len(labels), hs, 500) for _ in range(3)])
    #     best_losses.append((hs, np.mean(fixed_hs_losses[:, 0]), np.mean(fixed_hs_losses[:, 1])))
    #
    #     pprint(best_losses)
    #
    #     with open(os.path.join(DATA_FOLDER, "grid_search.json"), "wt") as f:
    #         json.dump(best_losses, f)

    train(np.array(x), np.array(y), len(labels), 64, 3000)