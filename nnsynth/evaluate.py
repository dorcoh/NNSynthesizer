"""
Provides an interface for plotting the network decision boundary,
a Dataset and trained NeuralNet objects are required
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from sklearn.metrics import accuracy_score


def get_grid_params(X, step_size=0.1):
    # TODO: remove, code duplication (in datasets.py)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    return xx, yy


def tr(elem):
    if elem == 0:
        return 'red'
    elif elem == 1:
        return 'blue'


class EvaluateDecisionBoundary:
    def __init__(self, clf, fixed_clf, dataset, meshgrid_stepsize, contourf_levels, save_plot):
        """required: dataset object `make` function has already been called"""
        self.clf = clf
        self.fixed_clf = fixed_clf
        self.dataset = dataset
        self.contourf_levels = contourf_levels
        self.save_plot = save_plot

        # get data and grid params
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataset.get_splitted_data()
        self.xx, self.yy = self.dataset.get_grid_params(meshgrid_stepsize)

    def plot(self, name='decision_boundary', use_test=False):
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(figsize=(20, 10))
        ax = plt.axes()

        if hasattr(self.clf, "decision_function"):
            Z = self.clf.decision_function(np.c_[self.xx.ravel(), self.yy.ravel()])
        else:
            Z = self.clf.predict_proba(np.c_[self.xx.ravel().astype(np.float32), self.yy.ravel().astype(np.float32)])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(self.xx.shape)
        ax.contourf(self.xx, self.yy, Z, self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

        # Plot the training points
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        if not use_test:
            ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                       edgecolors='c', alpha=0.6)

        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())

        ax.set_aspect('equal')
        # ax.grid(True, which='both')

        # plot accuracy score
        if use_test:
            y_true = self.clf.predict(self.X_test)
            acc_score = accuracy_score(y_true, self.y_test)
        else:
            y_true = self.clf.predict(self.X_train)
            acc_score = accuracy_score(y_true, self.y_train)

        ax.text(self.xx.max() - .3, self.yy.min() + .3, ('%.2f' % acc_score).lstrip('0'),
                size=15, horizontalalignment='right')

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

        ax.set_title(name)

        plt.tight_layout()

        if self.save_plot:
            plt.savefig(name + '.png')
        else:
            plt.show()

        # clean
        plt.delaxes(ax)

    def multi_plot(self, name='multi_plot', sub_name='sub_exp', split_sub_name=False, plot_train: bool = True, plot_test: bool = False):
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        plt.figure(figsize=(20, 10))

        fig, axes = plt.subplots(ncols=2)

        Z_dict = {0: None, 1: None}
        clfs_dict = {0: self.clf, 1: self.fixed_clf}
        for idx, ax in enumerate(axes):

            Z_dict[idx] = clfs_dict[idx].predict_proba(np.c_[self.xx.ravel().astype(np.float32),
                                                             self.yy.ravel().astype(np.float32)])[:, 1]

            # Put the result into a color plot
            Z_dict[idx] = Z_dict[idx].reshape(self.xx.shape)
            ax.contourf(self.xx, self.yy, Z_dict[idx], self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

            # Plot the training points
            # TODO: set inverse scaling (currently no scaling)
            if plot_train:
                ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                           edgecolors='k')
            # Plot the testing points
            if plot_test:
                ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                           edgecolors='c', alpha=0.6)

            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())

            ax.set_aspect('equal')
            # ax.grid(True, which='both')

            # plot accuracy score
            if plot_test:
                y_true = self.clf.predict(self.X_test)
                acc_score = accuracy_score(y_true, self.y_test)
            else:
                y_true = self.clf.predict(self.X_train)
                acc_score = accuracy_score(y_true, self.y_train)

            ax.text(self.xx.max() - .3, self.yy.min() + .3, ('%.2f' % acc_score).lstrip('0'),
                    size=15, horizontalalignment='right')

            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

            if idx == 0:
                ax.set_title("Original net")
            elif idx == 1:
                ax.set_title("Fixed net")

        plt.tight_layout()
        fig.suptitle("Exp: {}, Sub: {}".format(name, sub_name))

        if self.save_plot:
            if split_sub_name:
                base_dir = 'evaluator_results/' + name + '/'
                # TODO: change the hacky split
                plt.savefig(base_dir + sub_name.split(' ')[4] + '.png')
            else:
                plt.savefig(name + '.png')
        else:
            plt.show()

        plt.delaxes()


    def multi_plot_with_evalset(self, eval_set, threshold=None, name='multi_plot',
                                sub_name='sub_exp', split_sub_name=False):
        """Double plot (original vs. new), with evaluation set plotted as well
        accuracy calculated w.r.t eval set
        threshold - cuts the evaluation set as follows eval_set[:threshold]
        """
        X, y = eval_set[0][:threshold], eval_set[1][:threshold]

        # self.xx, self.yy = get_grid_params(X)

        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        plt.figure(figsize=(20, 10))

        fig, axes = plt.subplots(ncols=2)

        Z_dict = {0: None, 1: None}
        clfs_dict = {0: self.clf, 1: self.fixed_clf}
        print("X: \n{}".format(X))
        print("y: \n{}".format(y))
        for idx, ax in enumerate(axes):

            Z_dict[idx] = clfs_dict[idx].predict_proba(np.c_[self.xx.ravel().astype(np.float32),
                                                             self.yy.ravel().astype(np.float32)])[:, 1]

            # Put the result into a color plot
            Z_dict[idx] = Z_dict[idx].reshape(self.xx.shape)
            ax.contourf(self.xx, self.yy, Z_dict[idx], self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

            # Plot the training points (only for fixed net plot)
            # TODO: set inverse scaling (currently no scaling)

            colors = [i for i in map(tr, y.tolist())]
            if idx == 1:
                ax.scatter(X[:, 0], X[:, 1], c=colors, cmap=cm_bright, edgecolors='k', alpha=0.5)

            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())

            ax.set_aspect('equal')
            # ax.grid(True, which='both')

            # plot accuracy score
            y_pred_prob = clfs_dict[idx].predict_proba(X)
            y_pred = clfs_dict[idx].predict(X)
            print("Clf idx: {}".format(idx))
            print(y_pred_prob)
            print(y_pred)
            acc_score = accuracy_score(y_pred, y)

            ax.text(self.xx.max() - .3, self.yy.min() + .3, ('%.2f' % acc_score).lstrip('0'),
                    size=15, horizontalalignment='right')

            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

            if idx == 0:
                ax.set_title("Original net")
            elif idx == 1:
                ax.set_title("Fixed net")

        plt.tight_layout()
        fig.suptitle("Exp: {}, Sub: {}".format(name, sub_name))

        if self.save_plot:
            if split_sub_name:
                base_dir = 'evaluator_results/' + name + '/'
                if not Path(base_dir).exists():
                    Path(base_dir).mkdir()
                # TODO: change the hacky split
                file_path = base_dir + sub_name.split(' ')[4] + '.png'
                file_path = file_path.replace(',', '')
                print(file_path)
                plt.savefig(file_path)
            else:
                plt.savefig(name + '.png')
        else:
            plt.show()

        plt.delaxes()


    def multi_multi_plot(self, nets, name='multi_plot', plot_train: bool = True, plot_test: bool = False):
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        plt.figure(figsize=(20, 10))

        fig, axes = plt.subplots(ncols=len(nets))

        Z_dict = {i: None for i in range(len(nets))}
        clfs_dict = {i: nets[i] for i in range(len(nets))}
        for idx, ax in enumerate(axes):

            Z_dict[idx] = clfs_dict[idx].predict_proba(np.c_[self.xx.ravel().astype(np.float32),
                                                             self.yy.ravel().astype(np.float32)])[:, 1]

            # Put the result into a color plot
            Z_dict[idx] = Z_dict[idx].reshape(self.xx.shape)
            ax.contourf(self.xx, self.yy, Z_dict[idx], self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

            # Plot the training points
            # TODO: set inverse scaling (currently no scaling)
            if plot_train:
                ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                           edgecolors='k')
            # Plot the testing points
            if plot_test:
                ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                           edgecolors='c', alpha=0.6)

            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())

            ax.set_aspect('equal')
            # ax.grid(True, which='both')

            # plot accuracy score
            if plot_test:
                y_true = self.clf.predict(self.X_test)
                acc_score = accuracy_score(y_true, self.y_test)
            else:
                y_true = self.clf.predict(self.X_train)
                acc_score = accuracy_score(y_true, self.y_train)

            ax.text(self.xx.max() - .3, self.yy.min() + .3, ('%.2f' % acc_score).lstrip('0'),
                    size=15, horizontalalignment='right')

            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

            ax.set_title("Net {}".format(idx))

        plt.tight_layout()

        if self.save_plot:
            plt.savefig(name + '.png')
        else:
            plt.show()

        plt.delaxes()

    @classmethod
    def plot_patches(cls, patches: List, patches_labels: List):
        """Plots the grid patches"""
        fig, ax = plt.subplots()

        polys = []
        for patch in patches:
            arr = np.array(patch)
            matplot_poly = Polygon(arr, facecolor='none', edgecolor='black',
                           linewidth=5, closed=True, joinstyle='round')
            polys.append(matplot_poly)

        colors = np.array(patches_labels)
        p = PatchCollection(polys, alpha=0.4, facecolors='none', cmap=plt.cm.RdBu)
        p.set_array(colors)
        ax.add_collection(p)

        plt.autoscale(True)
        plt.show()

    def multi_plot_with_patches(self, patches: List, patches_labels: List, exp_name: str = '', sub_name: str = '',
                                eval_set: Tuple = None):
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        plt.figure(figsize=(20, 10))

        fig, axes = plt.subplots(ncols=2)

        Z_dict = {0: None, 1: None}
        clfs_dict = {0: self.clf, 1: self.fixed_clf}

        for idx, ax in enumerate(axes):

            Z_dict[idx] = clfs_dict[idx].predict_proba(np.c_[self.xx.ravel().astype(np.float32),
                                                             self.yy.ravel().astype(np.float32)])[:, 1]

            # Put the result into a color plot
            Z_dict[idx] = Z_dict[idx].reshape(self.xx.shape)
            ax.contourf(self.xx, self.yy, Z_dict[idx], self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())

            ax.set_aspect('equal')
            # ax.grid(True, which='both')

            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

            if idx == 0:
                ax.set_title("Original net")

            elif idx == 1:
                # add patches
                polys = []
                for patch in patches:
                    arr = np.array(patch)
                    matplot_poly = Polygon(arr, edgecolor='black',
                                           linewidth=1.5, closed=True, joinstyle='round')
                    polys.append(matplot_poly)

                colors = np.array(patches_labels)
                # where_0 = np.where(colors == 0)
                # where_1 = np.where(colors == 1)
                #
                # colors[where_0] = 1
                # colors[where_1] = 0

                # p = PatchCollection(polys, alpha=0.4, facecolors='none', cmap=plt.cm.RdBu, linewidths=2)
                p = PatchCollection(polys, cmap=cm_bright, alpha=0.25, match_original=True)
                p.set_array(colors)
                ax.add_collection(p, autolim=False)

                if eval_set:
                    X, y = eval_set[0], eval_set[1]
                    colors = [i for i in map(tr, y.tolist())]
                    ax.scatter(X[:, 0], X[:, 1], c=colors, cmap=cm_bright, edgecolors='k', alpha=0.5)

                ax.set_title("Fixed net")

        # plt.tight_layout()
        fig.suptitle("Exp: {}, Sub: {}".format(exp_name, sub_name))
        plt.autoscale(True)
        plt.show()