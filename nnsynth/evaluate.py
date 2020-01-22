"""
Provides an interface for plotting the network decision boundary,
a Dataset and trained NeuralNet objects are required
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


class EvaluateDecisionBoundary:
    def __init__(self, clf, fixed_clf, dataset, meshgrid_stepsize, contourf_levels, save_plot):
        """
        required: dataset object `make` function has already been called
        """
        self.clf = clf
        self.fixed_clf = fixed_clf
        self.dataset = dataset
        self.contourf_levels = contourf_levels
        self.save_plot = save_plot

        # get data and grid params
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataset.get_splitted_data()
        self.xx, self.yy = self.dataset.get_grid_params(meshgrid_stepsize)
        pass

    def plot(self, name='decision_boundary'):
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.axes()

        if hasattr(self.clf, "decision_function"):
            Z = self.clf.decision_function(np.c_[self.xx.ravel(), self.yy.ravel()])
        else:
            Z = self.clf.predict_proba(np.c_[self.xx.ravel().astype(np.float32), self.yy.ravel().astype(np.float32)])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(self.xx.shape)
        ax.contourf(self.xx, self.yy, Z, self.contourf_levels, vmin=0, vmax=1, cmap=cm, alpha=0.8)

        # Plot the training points
        # TODO: set inverse scaling (currently no scaling)
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                   edgecolors='c', alpha=0.6)

        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())

        ax.set_aspect('equal')
        # ax.grid(True, which='both')

        # plot accuracy score
        y_true = self.clf.predict(self.X_test)
        acc_score = accuracy_score(y_true, self.y_test)
        ax.text(self.xx.max() - .3, self.yy.min() + .3, ('%.2f' % acc_score).lstrip('0'),
                size=15, horizontalalignment='right')

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

        ax.set_title(name)

        plt.tight_layout()

        if self.save_plot:
            plt.savefig(name + '.png')
        elif not self.save_plot:
            plt.show()

        # clean
        plt.delaxes(ax)
