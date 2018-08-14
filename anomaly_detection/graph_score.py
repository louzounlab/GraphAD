from time import strftime, gmtime
from scipy import spatial
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from os import path


class GraphScore:
    def __init__(self, beta_matrix, database_name):
        self._database_name = database_name
        self._beta_matrix = beta_matrix
        self._score = []
        self._score_calculated = False
        num_of_graphs, num_ftr = beta_matrix.shape
        self._scores = [0] * num_of_graphs
        self._calc_score()

    def _calc_score(self):
        raise NotImplementedError()

    def score_list(self):
        if not self._score_calculated:
            self._score_calculated = True
            self._calc_score()
        return self._scores


class KnnScore(GraphScore):
    def __init__(self, beta_matrix, k, database_name, context_split=1):
        self._split = context_split
        self._dMat = None
        self._k = k
        super(KnnScore, self).__init__(beta_matrix, database_name)

    def _calc_score(self):

        """
        we recieve a matrix of bk by rows (row1=b1)
        :return:list of tuples (param,vertex)
        """
        self._dMat = spatial.distance_matrix(self._beta_matrix, self._beta_matrix)
        self._dMat = self._dMat.astype(float)
        np.fill_diagonal(self._dMat, np.inf)  # diagonal is zeros
        dim, dim = self._dMat.shape

        interval = int(dim / self._split)
        from_graph = 0
        to_graph = interval - 1
        for graph_k in range(dim):
            if graph_k >= interval + 5000:
                from_graph = graph_k - interval
                to_graph = graph_k
            sorted_row = np.sort(np.asarray(self._dMat[graph_k, from_graph:to_graph]))
            neighbor_sum = 0
            for col in range(self._k):
                neighbor_sum += sorted_row[col]
            self._scores[graph_k] = 1 / neighbor_sum

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


class KnnScore1(GraphScore):
    def __init__(self, beta_matrix, k, database_name, context_split=1):
        self._split = context_split
        self._k = k
        super(KnnScore, self).__init__(beta_matrix, database_name)

    def _calc_score(self):

        """
        we recieve a matrix of bk by rows (row1=b1)
        :return:list of tuples (param,vertex)
        """
        self._dMat = spatial.distance_matrix(self._beta_matrix, self._beta_matrix)
        self._dMat = self._dMat.astype(float)
        np.fill_diagonal(self._dMat, np.inf)  # diagonal is zeros

        num_graphs, dim = self._beta_matrix.shape

        interval = int(num_graphs / self._split)
        from_graph = 0
        to_graph = interval - 1
        for graph_k in range(dim):
            if graph_k >= interval:
                from_graph = graph_k - interval
                to_graph = graph_k

            current_beta = self._beta_matrix[graph_k]
            sub_matrix = self._beta_matrix[from_graph:to_graph, :]
            rows_average = np.average(sub_matrix, axis=0)
            sorted_ftr = np.argsort(np.abs(rows_average - current_beta))
            best_match_ftr = sorted_ftr[0:int(0.75 * len(sorted_ftr))]

            sub_matrix = sub_matrix.T[np.ix_(best_match_ftr)].T
            current_beta = current_beta.T[np.ix_(best_match_ftr)].T

            sorted_row = np.sort(np.asarray(spatial.distance_matrix(sub_matrix, np.matrix(current_beta)).T.tolist()[0])).tolist()
            # sorted_row = sorted_row[1:]
            neighbor_sum = 0
            for col in range(self._k):
                neighbor_sum += sorted_row[col]
            self._scores[graph_k] = 1 / neighbor_sum

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
