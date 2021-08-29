import unittest

import numpy as np
from cat_detector.bisecting_kmeans import L1_dist, assign_centroid, assign_points_to_cluster, gather_cluster_info, \
    LabelCounter, nsect_cluster, nsecting_kmeans


class TestBisectingKMeans(unittest.TestCase):

    def test_L1_dist(self):
        values = np.array([
            [5, 10],
            [5, 5],
            [3, 2],
            [8, 19]
        ], dtype="float32")

        centroids = np.zeros([2, 2], dtype="float32")
        centroids[0, :] = np.array([5, 10])
        centroids[1, :] = np.array([7, 18])

        l1_dist = L1_dist(values, centroids)

        self.assertListEqual([
            [0, 5, 10, 12],
            [10, 15, 20, 2]
        ], l1_dist.tolist())

    def test_assign_centroid(self):
        dists = np.array([
            [3, 5, 6, 7],
            [1, 7, 5, 4],
            [6, 8, 2, 0]
        ], dtype="float32")

        self.assertListEqual([1, 0, 2, 2], assign_centroid(dists).tolist())

    def test_assign_points_to_cluster(self):
        values = np.array([
            [5, 10],
            [5, 5],
            [3, 2],
            [8, 19]
        ], dtype="float32")

        labels = np.array([0, 1, 1, 1])

        centroids = np.zeros([2, 2], dtype="float32")
        centroids[0, :] = np.array([5, 10])
        centroids[1, :] = np.array([7, 18])

        new_labels = assign_points_to_cluster(values, labels, 1, centroids, np.array([2, 3]))

        self.assertListEqual([0, 2, 2, 3], new_labels.tolist())

    def test_gather_cluster_info(self):
        values = np.array([
            [5, 10],
            [5, 5],
            [2, 6],
            [3, 2],
            [8, 19]
        ], dtype="float32")

        labels = np.array([0, 0, 0, 1, 1])

        ci1 = gather_cluster_info(values, labels, 0)
        self.assertEqual(0, ci1.label)
        self.assertListEqual([4.0, 7.0], ci1.centroid.tolist())
        self.assertEqual(3, ci1.size)
        self.assertAlmostEqual(0.30303, ci1.avg_diff_from_mean, places=5)

        ci2 = gather_cluster_info(values, labels, 1)
        self.assertEqual(1, ci2.label)
        self.assertListEqual([5.5, 10.5], ci2.centroid.tolist())
        self.assertEqual(2, ci2.size)
        self.assertAlmostEqual(0.6875, ci2.avg_diff_from_mean, places=5)

    def test_nsect_cluster(self):
        values = np.array([
            [17, 16],
            [-13, -14],
            [5, 5],
            [2, 6],
            [3, 2],
            [18, 19],
            [-15, -17]
        ], dtype="float32")

        labels = np.zeros([values.shape[0]])

        counter = LabelCounter()

        np.random.seed(12345678)

        nsected_labels = nsect_cluster(values, labels, 0, counter, n=3)
        l1, l2, l3 = nsected_labels[0], nsected_labels[1], nsected_labels[2]

        self.assertEqual([l1, l2, l3, l3, l3, l1, l2], nsected_labels.tolist())

    def test_nsecting_kmeans(self):
        values = np.array([
            [7, 16],
            [-10, -11],
            [5, 5],
            [2, 6],
            [3, 2],
            [8, 19],
            [-15, -17]
        ], dtype="float32")

        for i in range(10):
            labels, infos = nsecting_kmeans(values,
                                            lambda clusters: sorted(clusters, key=lambda x: x.avg_diff_from_mean)[
                                                -1].label,
                                            lambda clusters: len(clusters) == 3, final_kmeans=True)

            l1, l2, l3 = labels[0], labels[1], labels[2]
            self.assertEqual([l1, l2, l3, l3, l3, l1, l2], labels.tolist())

            ci1 = [i for i in infos if i.label == l1][0]
            self.assertEqual(l1, ci1.label)
            self.assertListEqual([7.5, 17.5], ci1.centroid.tolist())
            self.assertEqual(2, ci1.size)
            self.assertAlmostEqual(0.08, ci1.avg_diff_from_mean, places=5)

            ci2 = [i for i in infos if i.label == l2][0]
            self.assertEqual(l2, ci2.label)
            self.assertListEqual([-12.5, -14], ci2.centroid.tolist())
            self.assertEqual(2, ci2.size)
            self.assertAlmostEqual(0.20755, ci2.avg_diff_from_mean, places=5)

            ci3 = [i for i in infos if i.label == l3][0]
            self.assertEqual(l3, ci3.label)
            self.assertAlmostEqual(3.33333, ci3.centroid.tolist()[0], places=5)
            self.assertAlmostEqual(4.33333, ci3.centroid.tolist()[1], places=5)
            self.assertEqual(3, ci3.size)
            self.assertAlmostEqual(0.34783, ci3.avg_diff_from_mean, places=5)
