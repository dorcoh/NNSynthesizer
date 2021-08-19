"""
Provides an interface for plotting the network decision boundary,
a Dataset and trained NeuralNet objects are required
"""
import logging
from pathlib import Path
from typing import List, Tuple, Union, Dict

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from sklearn.metrics import accuracy_score

from nnsynth.common.properties import DeltaRobustnessProperty, KeepContextType, KeepContextProperty


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


def build_exp_docstring(args, num_constraints, time_took, net_params, net_free_params):
    custom_exp_name = 'Soft' if args.soft_constraints else 'Hard'

    first = f"Repair by SMT :: Enforce {args.num_properties} properties \n" \
            f"Similarity Heuristic: {KeepContextType(args.heuristic).name} :: " \
            f"{num_constraints} {custom_exp_name} constraints :: " \
            f"Threshold {args.threshold}\n" \
            f"Solver time: {time_took} sec \n"

    second = f"NN details: Hidden layers sizes: {args.hidden_size}  \n" \
             f"# Parameters: {net_params}, Free: {net_free_params} \n" \
             f"Weights Selection: {args.weights_config}"

    fname = f"RepairResult::Props-{args.num_properties}::Heuristic-{KeepContextType(args.heuristic).name}::" \
            f"NumConstraints-{num_constraints}::Threshold-{args.threshold}::NNHidden-{args.hidden_size}::" \
            f"Params-{net_params}::Free-{net_free_params}::WeightsConfig-{args.weights_config}"

    return first, second, fname


def compute_exp_metrics(clf, fixed_clf, dataset, path: Union[None, Path] = None) -> Dict:
    logging.info("compute_exp_metrics")
    X_train, y_train, X_test, y_test = dataset.get_splitted_data()

    metrics = {}
    datasets = [('train', X_train, y_train), ('test', X_test, y_test), ('sampled', *dataset.get_sampled())]

    for clf_name, _clf in [('original', clf), ('repaired', fixed_clf)]:
        for set_name, X, y in datasets:
            y_pred = _clf.predict(X)
            n = y.shape[0]
            acc_score = accuracy_score(y, y_pred)
            key = clf_name + f"_{set_name}_acc"
            key_n = key + "_n"
            metrics[key] = acc_score
            metrics[key_n] = n

    orig_avgs = [value for key, value in metrics.items() if key.startswith('original') and not key.endswith('_n')]
    orig_weights = [value for key, value in metrics.items() if key.startswith('original') and key.endswith('_n')]
    repair_avgs = [value for key, value in metrics.items() if key.startswith('repaired') and not key.endswith('_n')]
    repair_weights = [value for key, value in metrics.items() if key.startswith('repaired') and key.endswith('_n')]

    metrics['original_avg'] = np.average(orig_avgs, weights=orig_weights)
    metrics['repaired_avg'] = np.average(repair_avgs, weights=repair_weights)

    if path:
        def remove_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix):]
        orig_metrics = {remove_prefix(key, 'original_'): value for key, value in metrics.items() if key.startswith('original')}
        repair_metrics = {remove_prefix(key, 'repaired_'): value for key, value in metrics.items() if key.startswith('repaired')}
        df = pd.DataFrame([pd.Series(orig_metrics).rename('original'), pd.Series(repair_metrics).rename('repaired')]).T
        df.to_csv(path, index=True)

    return metrics


class EvaluateDecisionBoundary:
    def __init__(self, clf, fixed_clf, dataset, meshgrid_stepsize, contourf_levels, save_plot, meshgrid_limit=.5,
                 x_limit=None, y_limit=None):
        """required: dataset object `make` function has already been called"""
        self.clf = clf
        self.fixed_clf = fixed_clf
        self.dataset = dataset
        self.contourf_levels = contourf_levels
        self.save_plot = save_plot

        # get data and grid params
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataset.get_splitted_data()
        self.xx, self.yy = self.dataset.get_grid_params(meshgrid_stepsize, meshgrid_limit)

        # limits can come from config or from decision boundary (default)
        self.x_limit = x_limit
        self.y_limit = y_limit
        x_lb = self.x_limit[0] if self.x_limit is not None else self.xx.min()
        x_ub = self.x_limit[1] if self.x_limit is not None else self.xx.max()
        self.x_limit = (x_lb, x_ub)
        y_lb = self.y_limit[0] if self.y_limit is not None else self.yy.min()
        y_ub = self.y_limit[1] if self.y_limit is not None else self.yy.max()
        self.y_limit = (y_lb, y_ub)

    def plot(self, name='decision_boundary', use_test=False):
        """Plot a NN decision boundary (no need to init self.fixed_clf)"""
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

    def plot_with_prop(self, property: List[DeltaRobustnessProperty], name='decision_boundary',
                       use_test: Union[None, bool] = None, path: Path = Path('.')):
        """Plot a NN decision boundary (no need to init self.fixed_clf)"""
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(figsize=(20, 10))
        ax = plt.axes()

        if hasattr(self.clf, "decision_function"):
            Z = self.clf.decision_function(np.c_[self.xx.ravel(), self.yy.ravel()])
        else:
            Z = self.clf.predict_proba(
                np.c_[self.xx.ravel().astype(np.float32), self.yy.ravel().astype(np.float32)])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(self.xx.shape)
        ax.contourf(self.xx, self.yy, Z, self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

        # Plot the training points
        if use_test is not None:
            if not use_test:
                ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                           edgecolors='k')
            # Plot the testing points
            else:
                ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                           edgecolors='c', alpha=0.6)

        ax.set_xlim(self.x_limit[0], self.x_limit[1])
        ax.set_ylim(self.y_limit[0], self.y_limit[1])

        ax.set_aspect('equal')
        # ax.grid(True, which='both')

        # plot accuracy score
        if use_test is not None:
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

        def gen_patches():
            # out of DeltaRobustnessProperty
            patches = []
            patches_labels = []
            for prop in property:
                curr_patch = []
                curr_patch.append((prop.coordinate[0] + prop.delta, prop.coordinate[1] + prop.delta))
                curr_patch.append((prop.coordinate[0] + prop.delta, prop.coordinate[1] - prop.delta))
                curr_patch.append((prop.coordinate[0] - prop.delta, prop.coordinate[1] - prop.delta))
                curr_patch.append((prop.coordinate[0] - prop.delta, prop.coordinate[1] + prop.delta))

                patches.append(curr_patch)
                patches_labels.append(prop.desired_output - 1)

            return patches, patches_labels

        patches, patches_labels = gen_patches()
        # add patches
        polys = []
        for patch in patches:
            arr = np.array(patch)
            print(arr)
            matplot_poly = Polygon(arr, edgecolor='black',
                                   linewidth=1.5, closed=True, joinstyle='round')
            polys.append(matplot_poly)

        hex_colors = [mcolors.hex2color(cm_bright.colors[i]) for i in patches_labels]
        colors = np.array(hex_colors)
        p = PatchCollection(polys, cmap=cm_bright, alpha=0.3)
        p.set_facecolor(colors)
        ax.add_collection(p, autolim=False)

        plt.tight_layout()

        if self.save_plot:
            fname = name + '.png'
            plt.savefig(path / fname)
        else:
            plt.show()

        # clean
        plt.delaxes(ax)

    def multi_plot(self, eval_set, name='multi_plot', sub_name='sub_exp', split_sub_name=False, fname: Path = Path('.')):
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        plt.figure(figsize=(30, 15))

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
            if idx == 0:
                # ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                #            edgecolors='k')
                ax.scatter(eval_set[0][:, 0], eval_set[0][:, 1], c=eval_set[1], cmap=cm_bright,
                           edgecolors='k')
            # Plot the testing points
            if idx == 1:
                # eval set
                pass

                # opacity = 0.15
                # if hasattr(self.dataset, 'X_sampled'):
                #     # plot sampled set
                #     ax.scatter(self.dataset.X_sampled[:, 0], self.dataset.X_sampled[:, 1],
                #                c=self.dataset.y_sampled, cmap=cm_bright, edgecolors='c', alpha=opacity)
                # else:
                #     # plot test set
                #     ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                #                edgecolors='c', alpha=opacity)

            ax.set_xlim(self.xx.min(), self.xx.max())
            ax.set_ylim(self.yy.min(), self.yy.max())

            ax.set_aspect('equal')
            # ax.grid(True, which='both')

            # plot accuracy score

            y_pred = clfs_dict[idx].predict(self.X_test)
            n_test = self.y_test.shape[0]
            test_acc_score = accuracy_score(self.y_test, y_pred)

            y_pred = clfs_dict[idx].predict(self.X_train)
            n_train = self.y_train.shape[0]
            train_acc_score = accuracy_score(self.y_train, y_pred)

            y_pred = clfs_dict[idx].predict(eval_set[0])
            n_eval = eval_set[1].shape[0]
            eval_acc_score = accuracy_score(eval_set[1], y_pred)

            text_size = 9
            delta_y = 4
            start = 10
            align_left = 9

            ax.text(self.xx.max() - align_left, self.yy.min() - (start + delta_y*2),
                    ('Test: %.5f, N: %d' % (test_acc_score, n_test)).lstrip('0'),
                    size=text_size, horizontalalignment='right')


            ax.text(self.xx.max() - align_left, self.yy.min() - (start + delta_y*1),
                    ('Train: %.5f, N: %d' % (train_acc_score, n_train)).lstrip('0'),
                    size=text_size, horizontalalignment='right')

            ax.text(self.xx.max() - align_left, self.yy.min() - (start + delta_y*0),
                    ('Eval: %.5f, N: %d' % (eval_acc_score, n_eval)).lstrip('0'),
                    size=text_size, horizontalalignment='right')

            if hasattr(self.dataset, 'X_sampled'):
                y_pred = clfs_dict[idx].predict(self.dataset.X_sampled)
                n_sampled = self.dataset.y_sampled.shape[0]
                sampled_acc_score = accuracy_score(self.dataset.y_sampled, y_pred)
                ax.text(self.xx.max() - align_left, self.yy.min() - (start + delta_y * 3),
                        ('Sampled: %.5f, N: %d' % (sampled_acc_score, n_sampled)).lstrip('0'),
                        size=text_size, horizontalalignment='right')

            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

            if idx == 0:
                ax.set_title("Original net")
            elif idx == 1:
                ax.set_title("Fixed net")

        fig.suptitle("{} \n {}".format(name, sub_name), fontsize=7)

        if self.save_plot:
            if split_sub_name:
                base_dir = 'evaluator_results/' + name + '/'
                # TODO: change the hacky split
                plt.savefig(base_dir + sub_name.split(' ')[4] + '.png')
            else:
                plt.savefig(fname.with_suffix('.png'))
        else:
            #plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            # plt.subplots_adjust(top=0.86, bottom=0.05)
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
                                eval_set: Tuple = None, name: Path = Path('.')):
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

            if idx == 0:
                ax.set_title("Original net")
                # add patches
                polys = []
                for patch in patches:
                    arr = np.array(patch)
                    matplot_poly = Polygon(arr, edgecolor='black',
                                           linewidth=1.5, closed=True, joinstyle='round')
                    polys.append(matplot_poly)

                hex_colors = [mcolors.hex2color(cm_bright.colors[i]) for i in patches_labels]
                colors = np.array(hex_colors)
                p = PatchCollection(polys, cmap=cm_bright, alpha=0.2, match_original=True)
                p.set_facecolor(colors)
                ax.add_collection(p, autolim=False)

            elif idx == 1:
                ax.set_title("Fixed net")

            # plot accuracy score

            y_pred = clfs_dict[idx].predict(self.X_test)
            n_test = self.y_test.shape[0]
            test_acc_score = accuracy_score(self.y_test, y_pred)

            y_pred = clfs_dict[idx].predict(self.X_train)
            n_train = self.y_train.shape[0]
            train_acc_score = accuracy_score(self.y_train, y_pred)

            text_size = 9
            delta_y = 4
            start = 10
            align_left = 9

            ax.text(self.xx.max() - align_left, self.yy.min() - (start + delta_y*2),
                    ('Test: %.5f, N: %d' % (test_acc_score, n_test)).lstrip('0'),
                    size=text_size, horizontalalignment='right')

            ax.text(self.xx.max() - align_left, self.yy.min() - (start + delta_y*1),
                    ('Train: %.5f, N: %d' % (train_acc_score, n_train)).lstrip('0'),
                    size=text_size, horizontalalignment='right')

            if hasattr(self.dataset, 'X_sampled'):
                y_pred = clfs_dict[idx].predict(self.dataset.X_sampled)
                n_sampled = self.dataset.y_sampled.shape[0]
                sampled_acc_score = accuracy_score(self.dataset.y_sampled, y_pred)
                ax.text(self.xx.max() - align_left, self.yy.min() - (start + delta_y * 3),
                        ('Sampled: %.5f, N: %d' % (sampled_acc_score, n_sampled)).lstrip('0'),
                        size=text_size, horizontalalignment='right')

            ax.set_aspect('equal')
            # ax.grid(True, which='both')

            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

        fig.suptitle("{} \n {}".format(exp_name, sub_name), fontsize=7)



        if self.save_plot:
            plt.savefig(name.with_suffix('.png'))
        else:
            plt.tight_layout(rect=(0, 0, 1, 0.95))
            plt.show()

    def multi_plot_all_heuristics(self,
                                  main_details: str,
                                  extra_details: str,
                                  keep_ctx_property: KeepContextProperty,
                                  metrics: Dict[str, float],
                                  path: Path,
                                  fname: str):
        """Generic function to evaluate all types of heuristics."""
        # colors
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        # fig
        plt.figure(figsize=(30, 15))
        fig, axes = plt.subplots(ncols=2)

        # decision boundary
        Z_dict = {0: None, 1: None}
        clfs_dict = {0: self.clf, 1: self.fixed_clf}

        # plotting
        for idx, ax in enumerate(axes):

            # calc decision boundary
            Z_dict[idx] = clfs_dict[idx].predict_proba(np.c_[self.xx.ravel().astype(np.float32),
                                                             self.yy.ravel().astype(np.float32)])[:, 1]
            Z_dict[idx] = Z_dict[idx].reshape(self.xx.shape)
            ax.contourf(self.xx, self.yy, Z_dict[idx], self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

            if idx == 0:
                ax.set_title("Original net")
                self.annotate_scores(ax, metrics, "original_")
                self.plot_heuristic_visualization(ax=ax, keep_ctx_property=keep_ctx_property, colormap=cm_bright)

            if idx == 1:
                ax.set_title("Fixed net")
                self.annotate_scores(ax, metrics, "repaired_")
                # plot repaired net

            ax.set_xlim(self.x_limit[0], self.x_limit[1])
            ax.set_ylim(self.y_limit[0], self.y_limit[1])
            ax.set_aspect('equal')
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')

        fig.suptitle("{} \n {}".format(main_details, extra_details), fontsize=7)

        if self.save_plot:

            try:
                _fname = fname + ".png"
                full_path = path / _fname
                plt.savefig(full_path)
            except OSError as exc:
                if exc.errno == 36:
                    _fname = "too_long.png"
                    full_path = path / _fname
                    plt.savefig(full_path)
        else:
            plt.show()

        plt.delaxes()

    def annotate_scores(self, ax, metrics: Dict[str, float], clf_key: str):
        delta_y = -15
        start_y = -35
        start_x = 0

        ax.annotate(text=('Weighted Average: %.5f' % (metrics[clf_key + "avg"])).lstrip('0'),
                    xy=(start_x, start_y + delta_y * 3),
                    xycoords='axes points')

        ax.annotate(text=('Test: %.5f, N: %d' % (metrics[clf_key + "train_acc"], metrics[clf_key + "train_acc_n"])).lstrip('0'),
                    xy=(start_x, start_y + delta_y * 2),
                    xycoords='axes points')

        ax.annotate(text=('Train: %.5f, N: %d' % (metrics[clf_key + "test_acc"], metrics[clf_key + "train_acc_n"])).lstrip('0'),
                    xy=(start_x, start_y + delta_y * 1),
                    xycoords='axes points')

        ax.annotate(text=('Sampled: %.5f, N: %d' % (metrics[clf_key + "sampled_acc"], metrics[clf_key + "sampled_acc_n"])).lstrip('0'),
                    xy=(start_x, start_y + delta_y * 0),
                    xycoords='axes points')

    def plot_heuristic_visualization(self, ax, keep_ctx_property: KeepContextProperty, colormap, **kwargs):

        # plot eval set in case of Samples
        if keep_ctx_property.get_keep_context_type() in [KeepContextType.SAMPLES]:
            eval_set = keep_ctx_property.get_kwargs('eval_set')
            ax.scatter(eval_set[0][:, 0], eval_set[0][:, 1], c=eval_set[1], cmap=colormap, edgecolors='k')

        # plot patches in case of Grid / Voronoi
        if keep_ctx_property.get_keep_context_type() in [KeepContextType.GRID, KeepContextType.VORONOI]:
            # add patches
            patches, patches_labels = keep_ctx_property.get_patches()
            polys = []
            for patch in patches:
                arr = np.array(patch)
                matplot_poly = Polygon(arr, edgecolor='black',
                                       linewidth=1.5, closed=True, joinstyle='round')
                polys.append(matplot_poly)

            hex_colors = [mcolors.hex2color(colormap.colors[i]) for i in patches_labels]
            colors = np.array(hex_colors)
            p = PatchCollection(polys, cmap=colormap, alpha=0.2, match_original=True)
            p.set_facecolor(colors)
            ax.add_collection(p, autolim=False)