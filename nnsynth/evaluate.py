"""
Provides an interface for plotting the network decision boundary,
a Dataset and trained NeuralNet objects are required
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class EvaluateDecisionBoundary:
    def __init__(self, clf, dataset, meshgrid_stepsize):
        """
        required: dataset object `make` function has already been called
        """
        self.clf = clf
        self.dataset = dataset

        # get data and grid params
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataset.get_splitted_data()
        # self.X, self.y = self.dataset.get_data()
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
        # score = self.clf.score(self.X_test, self.y_test)
        ax.contourf(self.xx, self.yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright,
                   edgecolors='c', alpha=0.6)

        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        # ax.text(self.xx.max() - .3, self.yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #         size=15, horizontalalignment='right')
        ax.set_aspect('equal')
        ax.grid(True, which='both')

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')

        plt.tight_layout()
        plt.savefig(name + '.png')

        # clean
        plt.delaxes(ax)
