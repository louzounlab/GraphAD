import datetime

from bokeh.colors import HSL
from time import strftime, gmtime
import numpy as np
import os
from bokeh.models import Label, LegendItem, Legend
from loggers import PrintLogger, BaseLogger
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from os import path
from bokeh.plotting import figure, show, save
from bokeh.palettes import Greens, Reds


class AnomalyPicker:
    def __init__(self, graphs, scores_list, database_name, logger: BaseLogger = None):
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


class SimpleAnomalyPicker(AnomalyPicker):
    def __init__(self, graphs, scores_list, database_name, num_anomalies=None):
        scores_list = [abs(i) for i in scores_list]
        super(SimpleAnomalyPicker, self).__init__(graphs, scores_list, database_name)

        p = np.percentile(scores_list, 60)
        self._bar = np.sort(scores_list)[-num_anomalies - 1] if num_anomalies else 2*p

    def build(self, truth=None):
        # splited has average of window [i - interval, i]
        for i, score in enumerate(self._scores_list):
            if score > self._bar:
                self._anomalies.append(i)
        if truth is None:
            return
        x_axis = list(range(len(self._scores_list)))
        FN, TN, TP, FP = 0, 0, 0, 0
        for x in x_axis:
            if x in truth and x not in self._anomalies:  # FN
                FN += 1
            elif x not in truth and x not in self._anomalies:  # TN
                TN += 1
            elif x in truth and x in self._anomalies:  # TP
                TP += 1
            elif x not in truth and x in self._anomalies:  # FP
                FP += 1
        recall = TP / (TP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        specificity = TN / (TN + FP + 1e-6)
        F1 = 2 * TP / ((2 * TP) + FP + FN + 1e-6)
        return FN, TN, TP, FP, recall, precision, specificity, F1

    def plot_anomalies_bokeh(self, file_name="context_anomalies", truth=None, info_text=None):
        plt_path = path.join("gif", self._database_name, file_name + "_" + strftime("%d:%m:%y_%H:%M:%S", gmtime()) + ".jpg")
        if "gif" not in os.listdir("."):
            os.mkdir("gif")
        if self._database_name not in os.listdir(path.join("gif")):
            os.mkdir(path.join("gif", self._database_name))

        x_axis = list(range(len(self._scores_list)))
        self._scores_list = [i if i > -1000 else -1000 for i in self._scores_list]
        BAR_WIDTH = 0.2
        p = figure(plot_width=1000, plot_height=250, title="anomalies",
                   x_axis_label="date", y_axis_label="Local_Outlier_score")
        p.vbar(x=x_axis, top=self._scores_list, line_color='blue', width=BAR_WIDTH)
        p.line(x_axis, [self._bar] * len(x_axis), line_color='red')

        if truth:
            green = Greens[9]
            red = Reds[9]
            max_anomalies = np.log(max(truth.values())) + 1e-10
            FN, TN, TP, FP = 0, 0, 0, 0
            for x, y in zip(x_axis, self._scores_list):
                if x in truth and x not in self._anomalies:         # FN
                    fnr = p.vbar(x=x, top=y, color=red[- 3 - int((5 * np.log(truth[x]) / max_anomalies))],
                                 width=BAR_WIDTH)  #
                    FN += truth[x]
                elif x not in truth and x not in self._anomalies:   # TN
                    tnr = p.vbar(x=x, top=y, color='cornflowerblue', width=BAR_WIDTH)
                    TN += 1
                elif x in truth and x in self._anomalies:           # TP
                    tpr = p.vbar(x=x, top=y, color=green[-3 - int((5 * np.log(truth[x]) / max_anomalies))],
                                 width=BAR_WIDTH)  # 'green'
                    TP += truth[x]
                elif x not in truth and x in self._anomalies:       # FP
                    fpr = p.vbar(x=x, top=y, color='orange', width=BAR_WIDTH)
                    FP += 1

            recall = TP / (TP + FN + 1e-6)
            precision = TP / (TP + FP + 1e-6)
            specificity = TN / (TN + FP + 1e-6)
            F1 = 2 * TP / (2 * TP + FP + FN + 1e-6)

            new_legend = Legend(items=[
                LegendItem(label="TP = " + str(TP)),    # renderers=[tpr]),
                LegendItem(label="FP = " + str(FP)),    # renderers=[fpr]),
                LegendItem(label="TN = " + str(TN)),    # renderers=[tnr]),
                LegendItem(label="FN = " + str(FN)),    # renderers=[fnr]),
                LegendItem(label="recall = " + "{:1.2f}".format(recall)),
                LegendItem(label="precision = " + "{:1.2f}".format(precision)),
                LegendItem(label="specificity = " + "{:1.2f}".format(specificity)),
                LegendItem(label="F1 = " + "{:1.2f}".format(F1)),
            ])
            p.add_layout(new_legend, 'left')

        else:
            for x, y in zip(x_axis, self._scores_list):
                if x in self._anomalies:  # predict anomaly
                    an = p.scatter(x, y, color='green')

            new_legend = Legend(items=[
                LegendItem(label="Anomalies", renderers=[an]),
            ])
            p.add_layout(new_legend, 'left')

        show(p)