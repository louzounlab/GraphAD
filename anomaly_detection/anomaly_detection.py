import multi_grapg_features
from beta_calculator import LinearRegBetaCalculator, LinearContext
from feature_calculators import FeatureMeta
from features_picker import PearsonFeaturePicker
from graph_score import KnnScore, ContextAnomaliesPicker, TestScore
from graphs import Graphs
from loggers import PrintLogger
from vertices.attractor_basin import AttractorBasinCalculator
from vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from vertices.betweenness_centrality import BetweennessCentralityCalculator
from vertices.bfs_moments import BfsMomentsCalculator
from vertices.closeness_centrality import ClosenessCentralityCalculator
from vertices.communicability_betweenness_centrality import CommunicabilityBetweennessCentralityCalculator
from vertices.eccentricity import EccentricityCalculator
from vertices.fiedler_vector import FiedlerVectorCalculator
from vertices.flow import FlowCalculator
from vertices.general import GeneralCalculator
from vertices.k_core import KCoreCalculator
from vertices.load_centrality import LoadCentralityCalculator
from vertices.louvain import LouvainCalculator
from vertices.motifs import nth_nodes_motif
from vertices.multi_dimensional_scaling import MultiDimensionalScalingCalculator
from vertices.page_rank import PageRankCalculator

ANOMALY_DETECTION_FEATURES = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),
    "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
    "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
    "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    # "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
    #                                                       {"communicability"}),
    "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
    "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),
    "flow": FeatureMeta(FlowCalculator, {}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
    # Isn't OK - also in previous version
    # "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
    "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),
    "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    "multi_dimensional_scaling": FeatureMeta(MultiDimensionalScalingCalculator, {"mds"}),
    "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
    # "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
    # "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    # "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}
REBUILD_FEATURES = False
RE_PICK_FTR = False


class AnomalyDetection:
    def __init__(self):
        # pearson + linear_regression(simple) + KNN + context
        self._params = {
            'database': 'EnronInc',
            # 'database': 'mc2_vast12',
            # 'database': 'twitter_security',
            'files_path': "../../databases/EnronInc/EnronInc_by_day",
            # 'files_path': "../../databases/mc2_vast12/basic_by_minute",
            # 'files_path': "../../databases/twitter_security/data_by_days",
            'date_format': '%d-%b-%Y.txt',  # Enron
            # 'date_format': '%d:%m:%Y_%H:%M.txt',  # vast
            # 'date_format': '%d:%m.txt',  # Twitter
            'directed': True,
            'max_connected': True,
            'logger_name': "default Anomaly logger",
            'number_of_feature_pairs': 10,
            'identical_bar': 0.9,
            'single_c': True,
            'dist_mat_file_name': "EnronInc_dist_test",
            'anomalies_file_name': "EnronInc_anomalies_test",
            'context_beta': 5,
            'KNN_k': 30,
            'context_split': 4,
            'context_bar': 0.3
        }
        self._ground_truth = [140, 330, 332, 352, 338, 393, 396]

        self._logger = PrintLogger("default Anomaly logger")
        self._graphs = Graphs(self._params['database'], files_path=self._params['files_path'], logger=self._logger,
                              features_meta=ANOMALY_DETECTION_FEATURES, directed=self._params['directed'],
                              date_format=self._params['date_format'], largest_cc=self._params['max_connected'])

    def build(self):
        self._graphs.build(force_rebuild_ftr=REBUILD_FEATURES, pick_ftr=RE_PICK_FTR)
        pearson_picker = PearsonFeaturePicker(self._graphs, size=self._params['number_of_feature_pairs'],
                                              logger=self._logger, identical_bar=self._params['identical_bar'])
        best_pairs = pearson_picker.best_pairs()
        # beta = LinearRegBetaCalculator(self._graphs, best_pairs, single_c=self._params['single_c'])
        beta = LinearContext(self._graphs, best_pairs, split=self._params['context_beta'])
        beta_matrix = beta.beta_matrix()
        score = KnnScore(beta_matrix, self._params['KNN_k'], self._params['database'],
                         context_split=self._params['context_beta'])
        # score = TestScore(beta_matrix, self._params['database'])
        score.dist_heat_map(self._params['dist_mat_file_name'])
        anomaly_picker = ContextAnomaliesPicker(self._graphs, score.score_list(), self._params['database'], logger=None,
                                                split=self._params['context_split'], bar=self._params['context_bar'])
        anomaly_picker.build()
        anomaly_picker.plot_anomalies(self._params['anomalies_file_name'], truth=self._ground_truth)


if __name__ == "__main__":
    AnomalyDetection().build()
