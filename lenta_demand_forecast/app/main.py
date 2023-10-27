import os
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from config import DATA_PATH, models, RANDOM_STATE
from model import Preprocessor, Optimizer, Predictor, types_changing

def pipeline(df: pd.DataFrame, means: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    '''Пайплайн для обработки каждого сгруппированного датафрейма отдельной моделью'''
    # Оптимизация моделей
    opt_res = {}
    opt_res['models'] = [model.__class__.__name__ for model in models]
    for model in models:
        opt = Optimizer(df, model, cat_features, means)
        metric, params = opt.tuning_optimize(trials=20)
        lags = params['lags']
        del params['lags']
        opt_res[model.__class__.__name__] = ([metric, params, lags])
    # Сохранение лучшей модели и её параметров
    best_res = 1000
    for model in opt_res['models']:
        if opt_res[model][0] < best_res:
            best_res = opt_res[model][0]
            best_model = model
    model_name = best_model
    model_params = opt_res[model][1]
    model_lags = opt_res[model][2]

    # Выполнение предобработки данных
    preprocessor = Preprocessor(df, means)
    future = preprocessor.future_preprocess()
    future.to_csv(os.path.join(DATA_PATH, "preprocessed/future_df.csv"), index=False)
    # Приведение типов данных к правильным
    future = pd.read_csv(os.path.join(DATA_PATH, "preprocessed/future_df.csv"), parse_dates=['ds'])
    types_changing(future, CAT_FEATURES)

    # Установка параметров модели
    if model_name == 'CatBoostRegressor':
        del model_params['min_child_weight']
        del model_params['colsample_bytree']
        model = cb.CatBoostRegressor(**model_params, random_state=RANDOM_STATE, cat_features=cat_features)
    elif model_name == 'XGBRegressor':
        model = xgb.XGBRegressor(**model_params, random_state=RANDOM_STATE, 
                        enable_categorical=True, tree_method='hist')
    elif model_name == 'LGBMRegressor':
        model = lgb.LGBMRegressor(**model_params, random_state=RANDOM_STATE)
    
    predictor = Predictor(df, future, model)
    preds = predictor.make_predictions()
    return preds


########### ЗДЕСЬ ДОЛЖНО БЫТЬ ПРИНЯТИЕ csv ФАЙЛОВ И СОХРАНЕНИЕ ИХ ПО ПУТИ os.path.join(DATA_PATH, file_name) ###########

# Выполнение предобработки данных
preprocessor = Preprocessor()
filled_df, means, CAT_FEATURES, EXOG_VARS = preprocessor.train_preprocess()

# Выборка данных для моделей. Можно заменить на любую другую выборку.
# groups = [filled_df.loc[(filled_df['pr_sku_id'] == '00661699f543753ec7e911a64b9fd2f6') \
#                        & (filled_df['st_id'] == '16a5cdae362b8d27a1d8f8c7b78b4330')]]
groups = [filled_df]

predictions = []
for df_test in groups:
    predictions.append(pipeline(df_test, means, CAT_FEATURES))

# Сохранение всех предсказаний
result = pd.concat(predictions)
result.to_csv('predictions.csv')