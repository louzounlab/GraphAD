import os
import pickle

from bokeh.models import Plot

from features_picker import PearsonFeaturePicker
from features_processor import FeaturesProcessor, log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
from parameters import AdParams, EnronParams, TwitterSecurityParams
from temporal_graph import TemporalGraph
from bokeh.layouts import column
from bokeh.palettes import cividis
from bokeh.io import export_png
from bokeh.plotting import figure, show, save
import os
import pickle
import numpy as np
import pandas as pd
# np.seterr(all='raise')
from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetStat:
    def __init__(self, params: AdParams):
        self._index_ftr = None
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._data_path = os.path.join(self._base_dir, "INPUT_DATA", params.database.DATABASE_FILE)
        self._params = params
        self._ground_truth = params.database.GROUND_TRUTH
        self._data_name = params.database.DATABASE_NAME
        self._logger = PrintLogger("Anomaly logger")
        self._temporal_graph = self._build_temporal_graph()
        # self._temporal_graph.filter(
        #         lambda x: False if self._temporal_graph.node_count(x) < 20 else True,
        #         func_input="graph_name")
        self._idx_to_name = list(self._temporal_graph.graph_names())
        self._name_to_idx = {name: idx for idx, name in enumerate(self._idx_to_name)}
        self._graph_to_vec = self._calc_vec()

    def _build_temporal_graph(self):
        database_name = self._data_name + "_" + str(self._params.max_connected) + "_" + str(
            self._params.directed)
        vec_pkl_path = os.path.join(self._base_dir, "pkl", "temporal_graphs", database_name + "_tg.pkl")
        if os.path.exists(vec_pkl_path):
            self._logger.info("loading pkl file - temoral_graphs")
            tg = pickle.load(open(vec_pkl_path, "rb"))
        else:
            tg = TemporalGraph(database_name, self._data_path, self._params.database.DATE_FORMAT,
                               self._params.database.TIME_COL, self._params.database.SRC_COL,
                               self._params.database.DST_COL, weight_col=self._params.database.WEIGHT_COL,
                               weeks=self._params.database.WEEK_SPLIT, days=self._params.database.DAY_SPLIT,
                               hours=self._params.database.HOUR_SPLIT, minutes=self._params.database.MIN_SPLIT,
                               seconds=self._params.database.SEC_SPLIT, directed=self._params.directed,
                               logger=self._logger).to_multi_graph()
            tg.suspend_logger()
            pickle.dump(tg, open(vec_pkl_path, "wb"))
        tg.wake_logger()
        return tg

    def _calc_vec(self):
        database_name = self._params.database.DATABASE_NAME + "_" + \
                        str(self._params.max_connected) + "_" + str(self._params.directed)
        vec_pkl_path = os.path.join(self._base_dir, "pkl", "vectors", database_name + "_vectors_log_" +
                                    str(self._params.log) + ".pkl")
        if os.path.exists(vec_pkl_path):
            self._logger.info("loading pkl file - graph_vectors")
            return pickle.load(open(vec_pkl_path, "rb"))

        gnx_to_vec = {}
        # create dir for database
        pkl_dir = os.path.join(self._base_dir, "pkl", "features")
        database_pkl_dir = os.path.join(pkl_dir, database_name)
        if database_name not in os.listdir(pkl_dir):
            os.mkdir(database_pkl_dir)

        for gnx_name, gnx in zip(self._temporal_graph.graph_names(), self._temporal_graph.graphs()):
            # create dir for specific graph features
            gnx_path = os.path.join(database_pkl_dir, gnx_name)
            if gnx_name not in os.listdir(database_pkl_dir):
                os.mkdir(gnx_path)

            gnx_ftr = GraphFeatures(gnx, self._params.features, dir_path=gnx_path, logger=self._logger,
                                    is_max_connected=self._params.max_connected)
            gnx_ftr.build(should_dump=True, force_build=self._params.FORCE_REBUILD_FEATURES)  # build features
            # calc motif ratio vector
            gnx_to_vec[gnx_name] = FeaturesProcessor(gnx_ftr).activate_motif_ratio_vec()

        pickle.dump(gnx_to_vec, open(vec_pkl_path, "wb"))
        return gnx_to_vec

    def _calc_matrix(self):
        database_name = self._data_name + "_" + str(self._params.max_connected) + "_" + str(
            self._params.directed)
        mat_pkl_path = os.path.join(self._base_dir, "pkl", "vectors", database_name + "_matrix.pkl")
        if os.path.exists(mat_pkl_path):
            self._logger.info("loading pkl file - graph_matrix")
            return pickle.load(open(mat_pkl_path, "rb"))

        gnx_to_vec = {}
        # create dir for database
        pkl_dir = os.path.join(self._base_dir, "pkl", "features")
        database_pkl_dir = os.path.join(pkl_dir, database_name)
        if database_name not in os.listdir(pkl_dir):
            os.mkdir(database_pkl_dir)

        for gnx_name, gnx in zip(self._temporal_graph.graph_names(), self._temporal_graph.graphs()):
            # create dir for specific graph features
            gnx_path = os.path.join(database_pkl_dir, gnx_name)
            if gnx_name not in os.listdir(database_pkl_dir):
                os.mkdir(gnx_path)

            gnx_ftr = GraphFeatures(gnx, self._params.features, dir_path=gnx_path, logger=self._logger,
                                    is_max_connected=self._params.max_connected)
            gnx_ftr.build(should_dump=True, force_build=self._params.FORCE_REBUILD_FEATURES)  # build features
            # calc motif ratio vector
            gnx_to_vec[gnx_name] = FeaturesProcessor(gnx_ftr).as_matrix()

        pickle.dump(gnx_to_vec, open(mat_pkl_path, "wb"))
        return gnx_to_vec

    # map matrix rows to features + count if there's more then one from feature
    def _set_index_to_ftr(self):
        gnx_name = self._temporal_graph.graph_names().__next__()
        gnx = self._temporal_graph.graphs().__next__()
        database_name = self._data_name + "_" + str(self._params.max_connected) + "_" + str(self._params.directed)
        gnx_path = os.path.join(self._base_dir, "pkl", "features", database_name, gnx_name)
        gnx_ftr = GraphFeatures(gnx, self._params.features, dir_path=gnx_path, logger=self._logger,
                                is_max_connected=self._params.max_connected)
        gnx_ftr.build(should_dump=False, force_build=self._params.FORCE_REBUILD_FEATURES)  # build features

        if not self._index_ftr:
            sorted_ftr = [f for f in sorted(gnx_ftr) if gnx_ftr[f].is_relevant()]  # fix feature order (names)
            self._index_ftr = []

            for ftr in sorted_ftr:
                len_ftr = len(gnx_ftr[ftr])
                # fill list with (ftr, counter)
                self._index_ftr += self._get_motif_type(ftr, len_ftr) if ftr == 'motif3' or ftr == 'motif4' else \
                    [(ftr, i) for i in range(len_ftr)]
        return self._index_ftr

    # return [ ... (motif_type, counter) ... ]
    def _get_motif_type(self, motif_type, num_motifs):
        header = []
        for i in range(num_motifs):
            header.append((motif_type, i))
        return header

    def plot_nodes_by_time(self):
        # collect data for plot
        nodes_count_by_time = self._temporal_graph.node_count()         # num of nodes per time
        edges_count_by_time = self._temporal_graph.edge_count()         # num of edges per time

        len_mg = self._temporal_graph.number_of_graphs()                # num of graphs (times)
        x_axis = list(range(len_mg))                                    # [0... num of times]

        p = figure(plot_width=600, plot_height=250, title=self._data_name + ", node & edge count",
                   x_axis_label="time", y_axis_label="nodes_count")                 # create figure

        p.line(x_axis, nodes_count_by_time, legend="nodes", line_color="blue")    # plot nodes
        p.line(x_axis, edges_count_by_time, legend="edges", line_color="green")  # plot edges

        # plot vertical lines for ground truth
        anomalies = [self._name_to_idx[anomaly] for anomaly in self._ground_truth]
        y = [edges_count_by_time[time] for time in anomalies]
        p.scatter(anomalies, y, legend="anomalies", line_color="red", fill_color="red")  # plot nodes
        p.xaxis.major_label_overrides = {i: graph_name for i, graph_name in
                                         enumerate(self._temporal_graph.graph_names())}     # time to graph_name dict
        p.legend.location = "top_left"
        show(p)

    def plot_timed_mean_std(self):
        NUM_PLOT_FTR = 20
        mat_dict = self._calc_matrix()
        ftrs = self._set_index_to_ftr()
        ftrs = [str(x) for x in ftrs]

        all_mx = np.vstack([mx for name, mx in mat_dict.items()])
        # sort by highest mean
        global_mean = {i: m for i, m in enumerate(np.mean(all_mx, 0).tolist()[0])}
        sorted_mean = [i for i, m in sorted(global_mean.items(), key=lambda x: -x[1])][0:NUM_PLOT_FTR]

        # ----------------------- mean -------------------------
        heat_mx = []
        mean_curves = [[] for i in range(NUM_PLOT_FTR)]
        std_curves = [[] for i in range(NUM_PLOT_FTR)]
        for name, mx in mat_dict.items():
            for i, idx in enumerate(sorted_mean):
                mx_mean = np.mean(mx, 0).tolist()[0]
                mx_std = np.std(mx, 0).tolist()[0]
                mean_curves[i].append(mx_mean[idx])
                std_curves[i].append(mx_std[idx])

        x_axis = list(range(self._temporal_graph.number_of_graphs()))  # [0... num of times]
        for i in range(1):#len(std_curves)):
            i = 16
            p = figure(plot_width=600, plot_height=250, title=self._data_name + " std/mean for " + ftrs[sorted_mean[i]],
                       x_axis_label="time", y_axis_label="nodes_count")  # create figure

            p.line(x_axis, mean_curves[i], legend="mean", line_color="blue")  # plot nodes
            p.line(x_axis, std_curves[i], legend="std", line_color="green")  # plot edges

            # plot vertical lines for ground truth
            anomalies = [self._name_to_idx[anomaly] for anomaly in self._ground_truth]
            y = [std_curves[i][time] for time in anomalies]
            p.scatter(anomalies, y, legend="anomalies", line_color="red", fill_color="red")  # plot nodes
            p.xaxis.major_label_overrides = {i: graph_name for i, graph_name in
                                             enumerate(self._temporal_graph.graph_names())}  # time to graph_name dict
            p.legend.location = "top_left"
            show(p)
            e = 0

    def plot_mean_std_sheatmap(self):
        ftrs = self._set_index_to_ftr()
        ftrs = [str(x) for x in ftrs]
        mat_dict = self._calc_matrix()
        # sort by highest std
        all_mx = np.vstack([mx for name, mx in mat_dict.items()])
        global_std = {i: m for i, m in enumerate(np.std(all_mx, 0).tolist()[0])}
        sorted_std = [i for i, m in sorted(global_std.items(), key=lambda x: -x[1])][0:30]

        # sort by highest mean
        global_mean = {i: m for i, m in enumerate(np.mean(all_mx, 0).tolist()[0])}
        sorted_mean = [i for i, m in sorted(global_mean.items(), key=lambda x: -x[1])][0:30]

        # global_max
        global_sum = {i: m for i, m in enumerate(np.max(all_mx, 0).tolist()[0])}

        anomalies = [self._name_to_idx[anomaly] for anomaly in self._ground_truth]

        # ----------------------- mean -------------------------
        heat_mx = []
        for name, mx in mat_dict.items():
            heat_day_mean = {i: m for i, m in enumerate(np.mean(mx, 0).tolist()[0])}
            heat_day_mean = [heat_day_mean[i] / global_sum[i] for i in sorted_mean]
            heat_mx.append(heat_day_mean)
        plt.subplots(figsize=(20, 15))
        heat_mx = np.vstack(heat_mx)
        ax = sns.heatmap(heat_mx, vmin=0.0005, vmax=0.005)
        plt.xticks(list(range(30)), ftrs[:30], rotation='vertical')
        for i in anomalies:
            ax.axhline(y=i, color='red', linewidth=0.4)
        plt.savefig("mean_heatmap")
        e = 0

        plt.clf()

        # ----------------------- std -------------------------
        heat_mx = []
        for name, mx in mat_dict.items():
            heat_day_std = {i: m for i, m in enumerate(np.std(mx, 0).tolist()[0])}
            heat_day_std = [heat_day_std[i] / global_sum[i] for i in sorted_std]
            heat_mx.append(heat_day_std)
        heat_mx = np.vstack(heat_mx)
        ax = sns.heatmap(heat_mx, vmin=0.005, vmax=0.05)
        plt.xticks(list(range(30)), ftrs[:30], rotation='vertical')
        for i in anomalies:
            ax.axhline(y=i, color='red', linewidth=0.4)
        plt.savefig("std_heatmap")
        e = 0

    def plot_features_mean_std(self):  # matrix: np.matrix):
        ftrs = self._set_index_to_ftr()
        ftrs = [str(x) for x in ftrs]

        #  -------------------- prepare matrix anomalies and rest of data
        all_list = []
        anomal_list = []
        for name, mx in self._calc_matrix().items():
            if name in self._ground_truth:
                anomal_list.append(mx)
            else:
                all_list.append(mx)

        all_mx = np.vstack(all_list)
        anomal_mx = np.vstack(anomal_list)

        global_mean = {i: m for i, m in enumerate(np.mean(all_mx, 0).tolist()[0])}
        global_max = {i: m for i, m in enumerate(np.std(all_mx, 0).tolist()[0])}
        sorted_keys = [i for i, m in sorted(global_mean.items(), key=lambda x: -x[1])]

        groups = []
        prev_val = global_max[sorted_keys[0]]
        sub_group = []
        size_ = 0
        for i in sorted_keys:
            if 100 * prev_val >= global_max[i] >= 00.1 * prev_val and size_ < 6:
                sub_group.append(i)
                size_ += 1
            else:
                prev_val = global_mean[i]
                groups.append(sub_group)
                sub_group = [i]
                size_ = 1

        for group_num in range(1):
            group_num = 2
            curr_ftr = []
            for i in groups[group_num]:
                curr_ftr.append(ftrs[i])
                curr_ftr.append("A_" + ftrs[i])
            mid = []
            bottom = []
            top = []
            for i in groups[group_num]:
                bottom.append(np.percentile(all_mx[:, i], 25, axis=0).tolist()[0])
                bottom.append(np.percentile(anomal_mx[:, i], 25, axis=0).tolist()[0])
                mid.append(np.percentile(all_mx[:, i], 50, axis=0).tolist()[0])
                mid.append(np.percentile(anomal_mx[:, i], 50, axis=0).tolist()[0])
                top.append(np.percentile(all_mx[:, i], 75, axis=0).tolist()[0])
                top.append(np.percentile(anomal_mx[:, i], 75, axis=0).tolist()[0])

            bottom = np.array(bottom)
            mid = np.array(mid)
            top = np.array(top)
            # find the quartiles and IQR for each category
            iqr = top - bottom
            upper = top + 1.5 * iqr
            lower = bottom - 1.5 * iqr

            p = figure(tools="", background_fill_color="#efefef", x_range=curr_ftr, toolbar_location=None,
                       plot_width=600, plot_height=600, title=self._data_name + "_percentile=(25-50-75)")

            colors = ["black", "red"] * int(mid.shape[0] / 2)
            # stems
            p.segment(curr_ftr, upper, curr_ftr, top, line_color=colors)
            p.segment(curr_ftr, lower, curr_ftr, bottom, line_color=colors)

            # boxes
            p.vbar(curr_ftr, 0.7, mid, top, fill_color="#E08E79", line_color=colors)
            p.vbar(curr_ftr, 0.7, bottom, mid, fill_color="#3B8686", line_color=colors)

            # whiskers (almost-0 height rects simpler than segments)
            p.rect(curr_ftr, lower, 0.2, 0.0000001, line_color=colors)
            p.rect(curr_ftr, upper, 0.2, 0.0000001, line_color=colors)

            p.xaxis.major_label_orientation = np.pi / 2
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = "white"
            p.grid.grid_line_width = 2
            p.xaxis.major_label_text_font_size = "12pt"
            show(p)
            # plot = Plot(output_backend="svg")
            # plot.output_backend(p, filename=str(group_num) + "_svg")

    def plot_correlations(self):
        from sklearn import linear_model
        mx_dict = self._calc_matrix()
        concat_mx = np.vstack([mx for name, mx in mx_dict.items()])
        pearson_picker = PearsonFeaturePicker(concat_mx, size=self._params.ftr_pairs,
                                              logger=self._logger, identical_bar=self._params.identical_bar)
        best_pairs = pearson_picker.best_pairs()
        for i, j, u in best_pairs:
            reg = linear_model.LinearRegression().fit(np.transpose(concat_mx[:, i].T), np.transpose(concat_mx[:, j].T))
            m = reg.coef_
            b = reg.intercept_

            ftr_i = concat_mx[:, i].T.tolist()[0]
            ftr_j = concat_mx[:, j].T.tolist()[0]

            p = figure(plot_width=600, plot_height=250, title=self._data_name + " regression " + str((i, j)),
                       x_axis_label="time", y_axis_label="nodes_count")  # create figure

            p.line(list(range(int(max(ftr_i)) + 1)), [m * i + b for i in range(10)], line_color="blue")  # plot nodes

            p.scatter(list(ftr_i), list(ftr_j))  # plot nodes
            p.xaxis.major_label_overrides = {i: graph_name for i, graph_name in
                                             enumerate(self._temporal_graph.graph_names())}  # time to graph_name dict
            p.legend.location = "top_left"
            show(p)

        e = 0


if __name__ == "__main__":
    ds = DatasetStat(AdParams(TwitterSecurityParams()))
    # ds.plot_timed_mean_std()
    # ds.plot_mean_std_sheatmap()
    # ds.plot_features_mean_std()
    # ds.plot_nodes_by_time()
    ds.plot_correlations()

