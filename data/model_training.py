from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

def train_random_forest(smote_train, smote_target):
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 1000,
        'max_features': 'sqrt',
        'max_depth': 4,
        'min_samples_leaf': 2,
        'random_state': 0,
        'verbose': 0
    }
    rf = RandomForestClassifier(**rf_params)
    rf.fit(smote_train, smote_target)
    return rf

def train_gradient_boosting(smote_train, smote_target):
    gb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 4,
        'min_samples_leaf': 2,
        'random_state': 0
    }
    gb = GradientBoostingClassifier(**gb_params)
    gb.fit(smote_train, smote_target)
    return gb

def train_xgboost(smote_train, smote_target):
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'random_state': 0
    }
    xgb_model = xgboost.XGBClassifier(**xgb_params)
    xgb_model.fit(smote_train, smote_target)
    return xgb_model
