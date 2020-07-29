import unittest
from pathlib import Path

import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from sympy import Plane, Point

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
        dataset.subset_data(0.001)
        self.X_train, self.y_train = dataset.get_data()

        # self.idx = np.random.randint(self.X_train.shape[0], size=int(0.05 * self.X_train.shape[0]))
        # self.vor = Voronoi(self.X_train[self.idx, :])
        # add reference points
        # self.points_set = np.append(self.X_train, [[20, 20], [-20, -20], [-20, 20], [20, -20]], axis=0)
        self.points_set = np.array([[0.5, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                                    [2, 0], [2, 1], [2, 2]])
        self.y_train = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
        self.vor = Voronoi(self.points_set)

    def test_something(self):
        """Plot Voronoi cells"""
        voronoi_plot_2d(self.vor)
        plt.show()

    def test_construct_voronoi_cells(self):
        """Plot Voronoi cells, and print all regions (infinite/finite)"""
        voronoi_plot_2d(self.vor)
        plt.show()
        print(self.vor.vertices)
        for region in self.vor.regions:
            print(region)
            reg_points = []
            for vert_idx in region:
                if vert_idx != -1:
                    v = self.vor.vertices[vert_idx]
                    reg_points.append(v)
            print(reg_points)

    def compute_eq_two_points(self, point_a, point_b):
        """Compute a linear equation out of two points, assumes 2d"""
        x1, y1 = point_a
        x2, y2 = point_b
        a = y1 - y2
        b = x2 - x1
        c = x1 * x2 - x2 * y1
        print("{} * x + {} * y + {} = 0".format(a, b, c))
        return a, b, c

    def calc_point_equations(self, equations, point):
        """Given equations and point, check the result of this point when evaluated on each of the equations"""
        x, y = point
        results = []
        for eq in equations:
            i, a, b, c = eq
            res = a*x + b*y + c
            results.append((i, res))

        return results

    def test_compute_equations(self):
        """Compute the equations of each region (currently takes one region),
        take two points and evaluate these equations with them"""
        polygons = []
        for region in self.vor.regions:
            reg_points = []
            if -1 in region:
                continue
            for vert_idx in region:
                v = self.vor.vertices[vert_idx]
                reg_points.append(v)
            if reg_points:
                polygons.append(reg_points)

        poly = polygons.pop()

        equations = []
        for i in range(len(poly)):
            point_a = poly[i]
            if i < len(poly)-1:
                point_b = poly[i+1]
            else:
                point_b = poly[0]

            a, b, c = self.compute_eq_two_points(point_a, point_b)
            equations.append((i, a, b, c))

        print(equations)
        results = self.calc_point_equations(equations, [1.0, 1.0])
        print(results)

        results = self.calc_point_equations(equations, [1.0, 2.0])
        print(results)

    def test_compute_plane(self):
        """Attempt to compute the plane out of the polygon coordinates"""
        polygons = []
        for region in self.vor.regions:
            reg_points = []
            if -1 in region:
                continue
            for vert_idx in region:
                v = self.vor.vertices[vert_idx]
                reg_points.append(v)
            if reg_points:
                polygons.append(reg_points)

        poly = polygons.pop()
        points = [Point(x[0], x[1]) for x in poly][:3]
        print(points)
        plane = Plane(*points)
        print(plane.equation())


    def test_print_poly(self):
        """Print the polygon coordinates"""
        polygons = []
        for region in self.vor.regions:
            reg_points = []
            if -1 in region:
                continue
            for vert_idx in region:
                v = self.vor.vertices[vert_idx]
                reg_points.append(v)
            if reg_points:
                polygons.append(reg_points)

        for poly in polygons:
            for point in poly:
                print(point)

    def test_something_else(self):
        for vpair in self.vor.ridge_points:
            if vpair[0] >= 0 and vpair[1] >= 0:
                v0 = self.vor.vertices[vpair[0]]
                v1 = self.vor.vertices[vpair[1]]
                # Draw a line from v0 to v1.
                plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)

        plt.show()

    def test_vor_regions(self):
        """Attempt to draw the calculated Voronoi regions"""
        fig, ax = plt.subplots()
        polygons = []
        patches = []
        for region in self.vor.regions:
            reg_points = []
            if -1 in region:
                continue
            for vert_idx in region:
                v = self.vor.vertices[vert_idx]
                reg_points.append(v)
            if reg_points:
                polygons.append(reg_points)

        for poly in polygons:
            mat_poly = Polygon(np.stack(poly), True)
            patches.append(mat_poly)

        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4, match_original=True)

        colors = np.empty(len(self.vor.point_region))
        for idx, region_idx in enumerate(self.vor.point_region):
            cls = self.y_train[idx]
            colors[region_idx - 1] = cls

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
        """Draw simple polygons (to test the lib)"""
        fig, ax = plt.subplots()
        patches = []
        num_polygons = 5
        num_sides = 5

        for i in range(num_polygons):
            polygon = Polygon(5 * np.random.rand(num_sides, 2), True)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

        colors = 100 * np.random.rand(len(patches))
        p.set_array(np.array(colors))

        ax.add_collection(p)

        ax.autoscale_view()
        plt.show()


if __name__ == '__main__':
    unittest.main()
