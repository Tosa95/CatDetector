from dataclasses import dataclass
from typing import Callable, Set, List, Iterable

import numpy as np


@dataclass
class ClusterInfo:
    label: int
    centroid: np.ndarray
    size: int
    avg_diff_from_mean: float


class LabelCounter:

    def __init__(self):
        self.count = 1

    def inc(self) -> int:
        res = self.count
        self.count += 1
        return res

    def get_all_labels(self) -> Set[int]:
        return set(range(self.count))


STOP_CRITERION = Callable[[List[ClusterInfo]], bool]
SELECT_CRITERION = Callable[[List[ClusterInfo]], int]


def L1_dist(values: np.ndarray, centroids: np.ndarray):
    values = np.expand_dims(values, 0)
    centroids = np.expand_dims(centroids, 1)

    ew_diff = np.abs(values - centroids)

    diff = np.sum(ew_diff, 2)

    return diff


def assign_centroid(dists: np.ndarray):
    dists = np.transpose(dists)
    return np.argmin(dists, 1)


def assign_points_to_cluster(values: np.ndarray, labels: np.ndarray, label_of_interest: int,
                             centroids: np.ndarray, centroids_labels: np.ndarray,
                             dist_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = L1_dist):
    labels = np.copy(labels)
    positions = np.argwhere(labels == label_of_interest).flatten()
    poi = values[positions]

    dists = dist_func(poi, centroids)
    new_centorids_indices = assign_centroid(dists)

    new_centroids_labels = centroids_labels[new_centorids_indices]

    labels[positions] = new_centroids_labels

    return labels


def assign_points_to_cluster_iterative(values: np.ndarray, labels: np.ndarray, label_of_interest: int,
                                       centroids: np.ndarray, centroids_labels: np.ndarray,
                                       dist_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = L1_dist,
                                       max_iter=10):
    old_labels = labels

    for i in range(max_iter):
        new_labels = assign_points_to_cluster(values, labels, label_of_interest, centroids, centroids_labels, dist_func)

        centroids = np.array([gather_cluster_info(values, new_labels, l).centroid for l in centroids_labels])

        if np.all(new_labels == old_labels) or i >= (max_iter - 1):
            if np.isnan(centroids).any():
                raise RuntimeError("Unsplittable")
            return new_labels

        old_labels = new_labels


def nsect_cluster(values: np.ndarray, labels: np.ndarray, label_of_interest: int,
                  counter: LabelCounter, dist_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = L1_dist,
                  max_iter: int = 10, n=2):

    poi = values[labels==label_of_interest]

    for i in range(10):
        centroids = poi[np.random.choice(range(poi.shape[0]), size=n, replace=False)]
        centroids_labels = np.array([counter.inc() for _ in range(n)])
        try:
            return assign_points_to_cluster_iterative(values, labels, label_of_interest, centroids, centroids_labels,
                                               dist_func, max_iter)
        except:
            pass
    raise RuntimeError("Unsplittable")


def nsecting_kmeans(values: np.ndarray,
                    select_criterion: SELECT_CRITERION,
                    stop_criterion: STOP_CRITERION,
                    dist_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = L1_dist,
                    max_iter: int = 10, n=2, final_kmeans=True):
    labels = np.zeros([values.shape[0]])
    counter = LabelCounter()
    removed_clusters = set()
    unsplittable_clusters = set()

    cluster_infos = gather_all_clusters_info(values, labels,
                                             labels_of_interest=counter.get_all_labels() - removed_clusters)

    while not stop_criterion(cluster_infos):
        cluster_to_nsect = select_criterion([c for c in cluster_infos if c.label not in unsplittable_clusters])
        try:
            labels = nsect_cluster(values, labels, cluster_to_nsect, counter, dist_func, max_iter, n)
            removed_clusters.add(cluster_to_nsect)
            cluster_infos = gather_all_clusters_info(values, labels,
                                                     labels_of_interest=counter.get_all_labels() - removed_clusters)
        except:
            unsplittable_clusters.add(cluster_to_nsect)

        if len(unsplittable_clusters) == len(cluster_infos):
            break


    if final_kmeans:
        labels = np.zeros([values.shape[0]])
        removed_clusters = set()
        centroids = np.array([ci.centroid for ci in cluster_infos])
        new_labels = np.array(range(1, len(centroids)+1))
        labels = assign_points_to_cluster_iterative(values, labels, 0, centroids, new_labels,
                                                    dist_func,
                                                    max_iter)
        cluster_infos = gather_all_clusters_info(values, labels,
                                                 labels_of_interest=new_labels)

    return labels, cluster_infos


def gather_all_clusters_info(values: np.ndarray, labels: np.ndarray, labels_of_interest: Iterable[int]) -> List[
    ClusterInfo]:
    return [gather_cluster_info(values, labels, l) for l in labels_of_interest]


def gather_cluster_info(values: np.ndarray, labels: np.ndarray, label_of_interest: int) -> ClusterInfo:
    poi = values[labels == label_of_interest]

    centroid = np.mean(poi, axis=0)
    size = poi.shape[0]
    diffs_from_mean = np.abs(poi - np.expand_dims(centroid, 0))
    avg_diff_from_mean = np.mean(diffs_from_mean) / np.mean(np.abs(centroid))

    return ClusterInfo(label=label_of_interest, centroid=centroid, size=size, avg_diff_from_mean=avg_diff_from_mean)
