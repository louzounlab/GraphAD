from time import strftime, gmtime
from scipy import spatial
import numpy as np
import seaborn as sns
import os
from graphs import Graphs
from loggers import PrintLogger, BaseLogger
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


class TestScore(GraphScore):
    def __init__(self, beta_matrix, database_name):
        self._dMat = None
        super(TestScore, self).__init__(beta_matrix, database_name)

    def _calc_score(self):

        """
        we recieve a matrix of bk by rows (row1=b1)
        :return:list of tuples (param,vertex)
        """
        num_of_graphs, num_ftr = self._beta_matrix.shape

        sum_ftr = sum(range(num_ftr))
        for graph_k in range(num_of_graphs):
            total = 0
            for i in range(num_ftr - 1, -1, -1):
                total += i*self._beta_matrix[graph_k, i]
            total /= sum_ftr
            if total == 0:
                total = 0.0001
            self._scores[graph_k] = 1 / total

        self._dMat = spatial.distance_matrix(np.matrix(self._scores).T, np.matrix(self._scores).T)
        self._dMat = self._dMat.astype(float)
        np.fill_diagonal(self._dMat, np.inf)  # diagonal is zeros

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
        for graph_k in range(dim):
            if graph_k < interval:
                sorted_row = np.sort(np.asarray(self._dMat[graph_k, 0:interval]))
            else:
                sorted_row = np.sort(np.asarray(self._dMat[graph_k, graph_k - interval:graph_k]))

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


class AnomaliesPicker:
    def __init__(self, graphs:Graphs, scores_list, database_name, logger: BaseLogger = None):
        self._database_name = database_name
        if logger:
            self._logger = logger
        else:
            self._logger = PrintLogger("default anomaly picker logger")
        self._graphs = graphs
        self._scores_list = scores_list
        self._anomalies = []
        self._anomalies_calculated = False

    def build(self):
        if not self._anomalies_calculated:
            self._anomalies_calculated = True
            self._calc()

    def anomalies_list(self):
        if not self._anomalies_calculated:
            self._anomalies_calculated = True
            self._calc()
        return self._anomalies

    def _calc(self):
        raise NotImplementedError()

    def plot_anomalies(self, file_name="anomalies"):
        raise NotImplementedError()


class ContextAnomaliesPicker(AnomaliesPicker):
    def __init__(self, graphs: Graphs, scores_list, database_name, logger: BaseLogger = None, split=4, bar=0.333):
        super(ContextAnomaliesPicker, self).__init__(graphs, scores_list, database_name)
        self._split = split
        self._bar = bar
        self._average_graph = []
        self._bar_graph = []

    def _calc(self):
        if len(self._scores_list) < self._split:
            self._logger.error("split number is bigger then number of graphs")
            return

        splited = []
        interval = int(len(self._scores_list) / self._split)
        for i in range(len(self._scores_list)):
            if i < interval:
                splited.append(np.average(self._scores_list[0:interval]))
            else:
                splited.append(np.average(self._scores_list[i-interval:i]))

        for avr, i in zip(splited, range(len(splited))):
            interval_bar = self._bar * avr
            self._average_graph.append(avr)
            self._bar_graph.append(interval_bar)
            if self._scores_list[i] < interval_bar:
                self._anomalies.append(i)

    def plot_anomalies(self, file_name="context_anomalies", truth=None, labels=5):
        plt_path = path.join("gif", self._database_name, file_name + "_" + strftime("%d:%m:%y_%H:%M:%S", gmtime()) + ".jpg")
        if "gif" not in os.listdir("."):
            os.mkdir("gif")
        if self._database_name not in os.listdir(path.join("gif")):
            os.mkdir(path.join("gif", self._database_name))

        x_axis = [x for x in range(len(self._scores_list))]
        y_axis = self._scores_list

        plt.plot(x_axis, self._average_graph)
        plt.plot(x_axis, self._bar_graph)

        plt.scatter(x_axis, y_axis, color='mediumaquamarine', marker="d", s=10)
        plt.title("parameter distribution")
        plt.xlabel("Time", fontsize=10)
        plt.ylabel("Graph", fontsize=10)

        # take 5 elements from x axis for display
        x_for_label = x_axis[::int(len(self._scores_list) / 5)]
        x_label = [self._graphs.index_to_name(x) for x in x_for_label]
        plt.xticks(x_for_label, x_label, rotation=3)
        patch = []
        if truth:
            for x, y in zip(x_axis, y_axis):
                if x in truth and x not in self._anomalies:  # false positive
                    plt.scatter(x, y, color='red', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='red'))
                elif x in truth and x in self._anomalies:  # true positive
                    plt.scatter(x, y, color='green', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='green'))
                elif x not in truth and x in self._anomalies:  # false negative
                    plt.scatter(x, y, color='black', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='black'))
                    # the is true negative
                plt.legend(handles=patch, fontsize='small', loc=2)
        else:
            for x, y in zip(x_axis, y_axis):
                if x in self._anomalies:  # predict anomaly
                    plt.scatter(x, y, color='green', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='green'))
                plt.legend(handles=patch, fontsize='small', loc=2)

        plt.savefig(plt_path)
        plt.clf()
        plt.close()
