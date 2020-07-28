import unittest
from pathlib import Path

import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull

from nnsynth.common.arguments_handler import ArgumentsParser
from nnsynth.datasets import XorDataset
import numpy as np
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        args = ArgumentsParser.parser.parse_args()
        print(args)
        # self.data = XorDataset(center=args.center, std=args.std, samples=args.dataset_size,
        #                      test_size=args.test_size, random_seed=args.random_seed)
        # self.X_train, self.y_train, X_test, y_test = self.data.get_splitted_data()
        data_path = Path('.').absolute().parent / 'good-network-data.XorDataset.pkl'
        dataset = XorDataset.from_pickle(data_path)
        dataset.subset_data(0.1)
        self.X_train, self.y_train, _, _ = dataset.get_splitted_data()

        # self.idx = np.random.randint(self.X_train.shape[0], size=int(0.05 * self.X_train.shape[0]))
        # self.vor = Voronoi(self.X_train[self.idx, :])
        # add reference points
        # self.points_set = np.append(self.X_train, [[20, 20], [-20, -20], [-20, 20], [20, -20]], axis=0)
        self.points_set = self.X_train
        self.vor = Voronoi(self.points_set)

    def test_something(self):
        voronoi_plot_2d(self.vor)
        plt.show()

    def test_something_else(self):
        for vpair in self.vor.ridge_points:
            if vpair[0] >= 0 and vpair[1] >= 0:
                v0 = self.vor.vertices[vpair[0]]
                v1 = self.vor.vertices[vpair[1]]
                # Draw a line from v0 to v1.
                plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)

        plt.show()

    def test_vor_regions(self):
        fig, ax = plt.subplots()
        polygons = []
        patches = []
        for region in self.vor.regions:
            reg_points = []
            for vert_idx in region:
                if vert_idx != -1:
                    v = self.vor.vertices[vert_idx]
                    reg_points.append(v)

            polygons.append(reg_points)

        for poly in polygons:
            if not poly:
                continue
            mat_poly = Polygon(np.stack(poly), True)
            patches.append(mat_poly)

        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4, match_original=True)

        colors = np.empty(len(self.vor.point_region))
        for idx, region_idx in enumerate(self.vor.point_region):
            cls = self.y_train[idx]
            colors[region_idx-1] = cls

        p.set_array(colors)
        ax.add_collection(p)
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax.scatter(self.points_set[:, 0], self.points_set[:, 1], c=self.y_train, cmap=cm_bright,
                   edgecolors='k')

        ax.autoscale_view()
        plt.show()

    def test_convex_hull(self):
        hull = ConvexHull(self.points_set[0, :])

    def test_plot_polys(self):
        fig, ax = plt.subplots()
        patches = []
        num_polygons = 5
        num_sides = 5

        for i in range(num_polygons):
            polygon = Polygon(5*np.random.rand(num_sides, 2), True)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

        colors = 100 * np.random.rand(len(patches))
        p.set_array(np.array(colors))

        ax.add_collection(p)

        ax.autoscale_view()
        plt.show()


if __name__ == '__main__':
    unittest.main()
