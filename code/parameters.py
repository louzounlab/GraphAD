from features_meta_ad import ANOMALY_DETECTION_FEATURES, MOTIF_FEATURES
import os


class AdParams:
    def __init__(self, database_params):
        self.database = database_params
        self.data_name = database_params.DATABASE_NAME
        self.n_outliers = database_params.GUESS if database_params.GUESS else \
            (len(database_params.GROUND_TRUTH) * 2 if database_params.GROUND_TRUTH else None)
        self.directed = True
        self.max_connected = False
        self.logger_name = "default Anomaly logger"

        # correlation
        self.ftr_pairs = 3
        self.identical_bar = 0.99
        # 'single_c': False,
        self.dist_mat_file_name = "dist_mat"
        self.anomalies_file_name = "anomalies"
        self.window_correlation = 10  # Enron

        # vectors representation
        self.vec_type = "regression"  # / "regression, motif_ratio
        self.log = True

        self.features = ANOMALY_DETECTION_FEATURES if self.vec_type == "regression" else MOTIF_FEATURES  #

        # score_method
        self.score_type = "gmm"  # / "gmm", "knn", "local_outlier"

        # GMM params
        self.n_components = 4

        # Local Outlier
        self.n_neighbors = 40

        # KNN params
        # 'context_beta': 4,
        self.KNN_k = 20  # Enron
        self.window_score = 50

        self.FORCE_REBUILD_FEATURES = False
        self._ignore_att = ["features", "logger_name", "database", "dist_mat_file_name", "anomalies_file_name",
                            "FORCE_REBUILD_FEATURES", "_ignore_att"]

    def attr_string(self):
        attr_str = []
        for attr in [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]:
            if attr in self._ignore_att:
                continue
            attr_str.append(attr)
        return ",".join(attr_str)

    def attr_val_string(self):
        attr_str = []
        for attr in [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]:
            if attr in self._ignore_att:
                continue
            attr_str.append(str(getattr(self, attr)))
        return ",".join(attr_str)

    def tostring(self):
        attr_str = []
        for attr in [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]:
            if attr in self._ignore_att:
                continue
            attr_str.append(attr + "," + str(getattr(self, attr)))
        return "\n".join(attr_str)


# ------------------------------------------------------ ENRON ---------------------------------------------------------
class EnronParams:
    def __init__(self):
        self.DATABASE_NAME = "Enron"
        self.DATABASE_FILE = "Enroninc.csv"
        self.GROUND_TRUTH = ['13-Dec-2000', '18-Oct-2001', '22-Oct-2001', '19-Nov-2001',
                             '23-Jan-2002', '30-Jan-2002', '04-Feb-2002']     # Enron
        self.GUESS = None
        self.DATE_FORMAT = "%d-%b-%Y"
        self.TIME_COL = "time"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.WEIGHT_COL = None
        self.WEEK_SPLIT = 0
        self.DAY_SPLIT = 1
        self.HOUR_SPLIT = 0
        self.MIN_SPLIT = 0
        self.SEC_SPLIT = 0


# ------------------------------------------------------ ENRON ---------------------------------------------------------
class OldEnronParams:
    def __init__(self):
        self.DATABASE_NAME = "Enron_Old"
        self.DATABASE_FILE = "Enroninc_Old.csv"
        self.GROUND_TRUTH = ['13-Dec-2000', '18-Oct-2001', '22-Oct-2001', '19-Nov-2001',
                             '23-Jan-2002', '30-Jan-2002', '04-Feb-2002']     # Enron
        self.GUESS = None
        self.DATE_FORMAT = "%d-%b-%Y"
        self.TIME_COL = "time"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.WEIGHT_COL = None
        self.WEEK_SPLIT = 0
        self.DAY_SPLIT = 1
        self.HOUR_SPLIT = 0
        self.MIN_SPLIT = 0
        self.SEC_SPLIT = 0


# ------------------------------------------------ twitter_security ----------------------------------------------------
class TwitterSecurityParams:
    def __init__(self):
        self.DATABASE_NAME = "Twitter_security"
        self.DATABASE_FILE = "Twitter_security.csv"
        self.GROUND_TRUTH = ['13_05', '20_05', '24_05', '30_05', '03_06', '05_06', '06_06', '09_06', '10_06', '11_06',
                             '15_06', '18_06', '19_06', '20_06', '25_06', '26_06', '03_07', '18_07', '30_07']  # Twitter
        self.GUESS = None
        self.DATE_FORMAT = "%d_%m"
        self.TIME_COL = "time"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.WEIGHT_COL = None
        self.WEEK_SPLIT = 0
        self.DAY_SPLIT = 1
        self.HOUR_SPLIT = 0
        self.MIN_SPLIT = 0
        self.SEC_SPLIT = 0


# --------------------------------------------------- mc2_vast12 -------------------------------------------------------
class MC2Vast12Params:
    def __init__(self):
        self.DATABASE_NAME = "mc2_vast12"
        self.DATABASE_FILE = "mc2_vast12"
        self.GROUND_TRUTH = ['4:5:2012_17:51', '4:5:2012_20:25', '4:5:2012_20:26', '4:5:2012_22:16', '4:5:2012_22:21'
                             , '4:5:2012_22:40', '4:5:2012_22:41', '4:6:2012_17:41', '4:5:2012_18:11']  # vast
        self.GUESS = None
        self.DATE_FORMAT = "%d:%m:%Y_%H:%M.txt"
        self.TIME_COL = "time"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.WEIGHT_COL = "weight"
        self.WEEK_SPLIT = 0
        self.DAY_SPLIT = 1
        self.HOUR_SPLIT = 0
        self.MIN_SPLIT = 0
        self.SEC_SPLIT = 0


# --------------------------------------------------- mc2_vast12 -------------------------------------------------------
#   - PageRank
class RealityMiningParams:
    def __init__(self):
        self.DATABASE_NAME = "Reality_Mining"
        self.DATABASE_FILE = "RealityMining.csv"
        self.GROUND_TRUTH = ["05-Sep-2004", "06-Sep-2004", "07-Sep-2004", "08-Sep-2004", "09-Sep-2004", "10-Sep-2004",
                             "11-Sep-2004",     # week 6
                             "17-Oct-2004", "18-Oct-2004", "19-Oct-2004", "20-Oct-2004", "21-Oct-2004", "22-Oct-2004",
                             "23-Oct-2004",     # week 12
                             "24-Oct-2004", "25-Oct-2004", "26-Oct-2004", "27-Oct-2004", "28-Oct-2004", "29-Oct-2004",
                             "30-Oct-2004",     # week 13
                             "07-Nov-2004", "08-Nov-2004", "09-Nov-2004", "10-Nov-2004", "11-Nov-2004", "12-Nov-2004",
                             "13-Nov-2004",     # week 15
                             "14-Nov-2004", "15-Nov-2004", "16-Nov-2004", "17-Nov-2004", "18-Nov-2004", "19-Nov-2004",
                             "20-Nov-2004",     # week 16
                             "21-Nov-2004", "22-Nov-2004", "23-Nov-2004", "24-Nov-2004", "25-Nov-2004", "26-Nov-2004",
                             "27-Nov-2004",     # week 17
                             "05-Dec-2004", "06-Dec-2004", "07-Dec-2004", "08-Dec-2004", "09-Dec-2004", "10-Dec-2004",
                             "11-Dec-2004",     # week 19
                             "12-Dec-2004", "13-Dec-2004", "14-Dec-2004", "15-Dec-2004", "16-Dec-2004", "17-Dec-2004",
                             "18-Dec-2004",     # week 20
                             "19-Dec-2004", "20-Dec-2004", "21-Dec-2004", "22-Dec-2004", "23-Dec-2004", "24-Dec-2004",
                             "25-Dec-2004",     # week 21
                             "26-Dec-2004", "27-Dec-2004", "28-Dec-2004", "29-Dec-2004", "30-Dec-2004", "31-Dec-2004",
                             "01-Jan-2005",     # week 22
                             "02-Jan-2005", "03-Jan-2005", "04-Jan-2005", "05-Jan-2005", "06-Jan-2005", "07-Jan-2005",
                             "08-Jan-2005",     # week 23
                             "30-Jan-2005", "31-Jan-2005", "01-Feb-2005", "02-Feb-2005", "03-Feb-2005", "04-Feb-2005",
                             "05-Feb-2005",     # week 27
                             "27-Feb-2005", "28-Feb-2005", "01-Mar-2005", "02-Mar-2005", "03-Mar-2005", "04-Mar-2005",
                             "05-Mar-2005",     # week 31
                             "06-Mar-2005", "07-Mar-2005", "08-Mar-2005", "09-Mar-2005", "10-Mar-2005", "11-Mar-2005",
                             "12-Mar-2005",     # week 32
                             "20-Mar-2005", "21-Mar-2005", "22-Mar-2005", "23-Mar-2005", "24-Mar-2005", "25-Mar-2005",
                             "26-Mar-2005",     # week 34
                             "27-Mar-2005", "28-Mar-2005", "29-Mar-2005", "30-Mar-2005", "31-Mar-2005", "01-Apr-2005",
                             "02-Apr-2005"]     # week 35  ground truth in weeks
        self.GUESS = None
        self.DATE_FORMAT = "%d-%b-%Y"
        self.TIME_COL = "time"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.WEIGHT_COL = None
        self.WEEK_SPLIT = 0
        self.DAY_SPLIT = 1
        self.HOUR_SPLIT = 0
        self.MIN_SPLIT = 0
        self.SEC_SPLIT = 0


# ------------------------------------------------------ ENRON ---------------------------------------------------------
class SecRepoParams:
    def __init__(self):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0])

        self.DATABASE_NAME = "SecRepo"
        self.DATABASE_FILE = "SecRepo_auth_log.csv"
        self.GROUND_TRUTH = self._load_ground_truth()
        self.GUESS = None
        self.DATE_FORMAT = "%b_%d_%H:%M:%S"
        self.TIME_COL = "time"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.WEIGHT_COL = None
        self.WEEK_SPLIT = 0
        self.DAY_SPLIT = 0
        self.HOUR_SPLIT = 1
        self.MIN_SPLIT = 0
        self.SEC_SPLIT = 0

    def _load_ground_truth(self):
        gt_dict = {}
        src = open(os.path.join(self._base_dir, "INPUT_DATA", "SecRepo_auth_log_ground_truth.csv"))
        for row in src:
            date, count = row.split(",")
            if int(count) > 10:
                gt_dict[date] = 1
        return gt_dict


DEFAULT_PARAMS = AdParams(EnronParams())
