from time import strftime, gmtime
from scipy import spatial
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from os import path
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import cosine, jaccard, directed_hausdorff, correlation


class GraphScore:
    def __init__(self, beta_matrix, database_name):
        self._database_name = database_name
        self._beta_matrix = beta_matrix
        self._score = []
        self._score_calculated = False
        self._num_of_graphs, self._num_ftr = beta_matrix.shape
        self._scores = [0] * self._num_of_graphs
        self._calc_score()

    def _calc_score(self):
        raise NotImplementedError()

    def score_list(self):
        if not self._score_calculated:
            self._score_calculated = True
            self._calc_score()
        return self._scores


class KnnScore(GraphScore):
    def __init__(self, beta_matrix, k, database_name, window_size):
        self._split = beta_matrix.shape[0] if window_size is None else window_size
        self._dMat = None
        self._k = k
        super(KnnScore, self).__init__(beta_matrix, database_name)

    def _calc_score(self):
        """
        we receive a matrix of bk by rows (row1=b1)
        :return:list of tuples (param,vertex)
        """
        self._dMat = spatial.distance_matrix(self._beta_matrix, self._beta_matrix)
        self._dMat = self._dMat.astype(float)
        np.fill_diagonal(self._dMat, np.inf)  # diagonal is zeros
        dim, dim = self._dMat.shape

        interval = self._split
        from_graph = 0
        to_graph = interval - 1
        for graph_k in range(dim):
            if graph_k >= interval:
                from_graph = graph_k - interval
                to_graph = graph_k
            sorted_row = np.sort(np.asarray(self._dMat[graph_k, from_graph:to_graph]))
            neighbor_sum = 1e-4
            for col in range(self._k):
                neighbor_sum += sorted_row[col]
            self._scores[graph_k] = neighbor_sum

    def dist_heat_map(self, file_name="graph_dist"):
        plt_path = path.join("gif", self._database_name, file_name + "_" + strftime("%d:%m:%y_%H:%M:%S", gmtime()) + ".jpg")
        if "gif" not in os.listdir("."):
            os.mkdir("gif")
        if self._database_name not in os.listdir(path.join("gif")):
            os.mkdir(path.join("gif", self._database_name))
        ax = sns.heatmap(self._dMat, xticklabels=40, yticklabels=40, vmin=0, vmax=1)
        plt.title("Graphs Distance")
        plt.savefig(plt_path)
        plt.close()


class GmmScore(GraphScore):
    def __init__(self, beta_matrix, database_name, window_size=None, n_components=5):
        self._split = beta_matrix.shape[0] if window_size is None else window_size
        self._gmm = GaussianMixture(covariance_type="diag", warm_start=True, n_components=n_components)
        super(GmmScore, self).__init__(beta_matrix, database_name)

    def _calc_score(self):
        num_graphs, num_ftr = self._beta_matrix.shape
        interval = self._split
        self._gmm.fit(self._beta_matrix[:interval - 1])

        for graph_k in range(num_graphs):
            if graph_k >= interval:
                from_graph = graph_k - interval
                to_graph = graph_k
                self._gmm.fit(self._beta_matrix[from_graph:to_graph])
            self._scores[graph_k] = self._gmm.score_samples([self._beta_matrix[graph_k]])[0]


class LocalOutlierFactorScore(GraphScore):
    def __init__(self, beta_matrix, database_name, window_size=None, n_neighbors=40):
        self._split = beta_matrix.shape[0] if window_size is None else window_size
        self._clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        super(LocalOutlierFactorScore, self).__init__(beta_matrix, database_name)

    def _calc_score(self):
        num_graphs, num_ftr = self._beta_matrix.shape
        interval = self._split
        self._clf.fit(self._beta_matrix[:interval - 1])

        for graph_k in range(num_graphs):
            if graph_k >= interval:
                from_graph = graph_k - interval
                to_graph = graph_k
                self._clf.fit(self._beta_matrix[from_graph:to_graph])
            self._scores[graph_k] = self._clf._decision_function([self._beta_matrix[graph_k]])[0]


