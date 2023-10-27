import os
import catboost as cb
import xgboost as xgb
import lightgbm as lgb

# Root path of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to texts with exercises and uploaded texts
DATA_PATH = os.path.join(ROOT_DIR, 'data/')

# Константы
RANDOM_STATE = 112233
# Количество дней в тестовой выборке
test_days = 14
# Количество фолдов в cv
n_folds = 3

# Список моделей для использования
models = [lgb.LGBMRegressor(silent=True)]#, xgb.XGBRegressor(silent=True), cb.CatBoostRegressor(silent=True)]