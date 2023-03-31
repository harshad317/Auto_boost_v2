# Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, precision_score, auc, recall_score
from sklearn.metrics import r2_score, log_loss, precision_recall_curve, make_scorer
from sklearn.feature_selection import SelectKBest
from boruta import BorutaPy
#from Borutashap import Borutashap