import os
import pandas as pd
import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import optuna
from prophet import Prophet
from etna.transforms import TimeSeriesImputerTransform, DensityOutliersTransform
from etna.datasets import TSDataset
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Dict, Tuple
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from mlforecast import MLForecast
from config import DATA_PATH, n_folds, test_days, RANDOM_STATE

# Отключим предупреждения
import warnings
warnings.filterwarnings("ignore")

def wape(target: pd.Series, predictions: pd.Series) -> float:
    '''Метрика WAPE для оценки моделей'''
    result = np.sum(np.abs(target - predictions)) / np.sum(np.abs(target))
    return result


def make_date_features(data: pd.DataFrame) -> pd.DataFrame:
    '''Функция для создания календарных признаков'''
    df = data.copy()
    # Добавим календарные признаки
    df['day'] = df['date'].dt.day.astype('category')
    df['month'] = df['date'].dt.month.astype('category')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_number'] = df['date'].dt.strftime('%U').astype('category')
    # Кодируем день недели тригонометрическими функциями
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df.drop('day_of_week', axis=1, inplace=True)

    return df

def types_changing(df: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    # Изменим тип данных для определенных столбцов, так как они категориальные по своей сути
    for col in ['st_type_format_id', 'st_type_loc_id', 'st_type_size_id', 'pr_uom_id']:
        df[col] = df[col].astype('category')
    df['day'] = df['day'].astype('category')
    df['month'] = df['month'].astype('category')
    df['week_number'] = df['week_number'].astype('category')
    df['holiday'] = df['holiday'].astype('category')
    for feature in cat_features:
            df[feature] = df[feature].astype('category')
    return df


class Preprocessor:
    '''Класс для предобработки данных перед использованием моделей.'''
    def __init__(self, df: pd.DataFrame = pd.DataFrame(), means: pd.DataFrame = pd.DataFrame()) -> None:
        # Данные о праздничных днях
        file_path = os.path.join(DATA_PATH, 'holidays_covid_calendar.csv')
        self.holidays = pd.read_csv(file_path)
        self.holidays = self.holidays[self.holidays['holiday'] == 1].copy()
        self.holidays['date'] = pd.to_datetime(self.holidays['date'], format='%d.%m.%Y')
        self.means = means
        self.df = df

    def baseline(self) -> pd.DataFrame:
        '''Базовая предобработка всего датафрейма'''

        # Прочитаем файлы 
        df = {}
        for file in ['sales_df_train', 'pr_df', 'st_df', 'sales_submission']:
            file_name = f"{file}.csv"
            file_path = os.path.join(DATA_PATH, file_name)
            df[file] = pd.read_csv(file_path)

        shops = df['sales_submission']['st_id'].unique()

        # Удалим строки для которых число проданных товаров = 0, но при этом столбец продажа в руб отличается от 0
        df['sales_df_train'] = df['sales_df_train'].loc[~((df['sales_df_train']['pr_sales_in_units'] == 0) & 
                                                        (df['sales_df_train']['pr_sales_in_rub'] != 0))]

        # Переведем столбец date в формат даты
        df['sales_df_train']['date'] = pd.to_datetime(df['sales_df_train']['date'], format='%Y-%m-%d')

        # Удалим строки для которых продажи не равны 0, но продажи в штуках отсутствуют
        df['sales_df_train'] = df['sales_df_train'].loc[~((df['sales_df_train']['pr_sales_in_rub'] == 0)&
                                                        (df['sales_df_train']['pr_sales_in_units'] != 0))]

        # Оставляем для дальнейшего анализа только активные магазины и те, которые есть в sales_submission
        df['sales_df_train'] = df['sales_df_train'].loc[df['sales_df_train']['st_id'].isin(shops)]

        # Объединяем sales_df_train и st_df
        data = df['sales_df_train'].merge(df['st_df'], how='left', on='st_id')
        data.head()

        # Объединяем data и pr_df
        data = data.merge(df['pr_df'], how='left', on='pr_sku_id')
        data.head()

        # Сортируем по магазину, товару и по дате
        data = data.sort_values(['st_id', 'pr_sku_id', 'date'])
        data.head()

        # Добавляем признаки общие продажи в штуках и рублях (промо + без промо)
        data['total_sales_in_units'] = data.groupby(['st_id', 'pr_sku_id', 'date'])['pr_sales_in_units'].transform('sum')
        data['total_sales_in_rub'] = data.groupby(['st_id', 'pr_sku_id', 'date'])['pr_sales_in_rub'].transform('sum')

        # Добавляем признак праздничного дня
        data['holiday'] = data['date'].isin(self.holidays['date']).astype('category')

        # Удалим строки у которых total_sales_in_units = 0, а в рублях продажи есть
        data = data.loc[~((data['total_sales_in_units'] == 0)&(data['total_sales_in_rub'] != 0))]

        # Создание новых ценовых признаков
        data['price'] = data['total_sales_in_rub'] / data['total_sales_in_units']

        # Удаляем ненужные столбцы и получившиеся дубликаты
        data = data.drop(['pr_sales_type_id', 'pr_sales_in_units', 'pr_promo_sales_in_units', 'pr_sales_in_rub',
                        'pr_promo_sales_in_rub', 'st_is_active', 'total_sales_in_rub'], axis = 1)
        data = data.drop_duplicates()

        # Изменим тип данных для определенных столбцов, так как они категориальные по своей сути
        for col in ['st_type_format_id', 'st_type_loc_id', 'st_type_size_id', 'pr_uom_id']:
            data[col] = data[col].astype('category')

        return data

    def fill_missing(self) -> pd.DataFrame:
        '''Заполнение пропусков временных рядов'''
        data = self.baseline()

        # Создаем DataFrame с диапазоном дат
        dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
        dates_df = pd.DataFrame({'date': dates})

        # Создаем пустой список для хранения заполненных данных
        filled_data = []

        # Создаём уникальный айди для каждого товара-магазина
        unique_sku_id = 0

        # Цикл для заполнения данных для каждого магазина и товара
        for store in data['st_id'].unique():
            for product in tqdm(data[data['st_id'] == store]['pr_sku_id'].unique()):
                # Получаем текущий фрагмент данных для конкретного магазина и товара
                subset = data[(data['st_id'] == store) & (data['pr_sku_id'] == product)]
                filling_row = subset.iloc[0]

                # приводим данные к ETNA-формату
                ts = subset[['date', 'total_sales_in_units', 'pr_sku_id']]
                ts.columns = ['timestamp', 'target', 'segment']
                ts = TSDataset.to_dataset(ts)
                ts = TSDataset(ts, freq="D")

                # outliers detection - Prediction interval method
                # Need at least 3 dates to infer frequency
                if len(subset) > 3:
                    outliers_remover = DensityOutliersTransform(in_column="target", window_size=14, distance_coef=1, n_neighbors=1)
                    ts.fit_transform([outliers_remover])

                # Impute NaNs using the specified strategy
                imputer = TimeSeriesImputerTransform(in_column="target", strategy="running_mean", window=7)
                ts = imputer.fit_transform(ts)
                ts = ts.to_pandas(flatten = True)
                ts.columns = ['date', 'pr_sku_id', 'total_sales_in_units']
                ts = ts.fillna(0)

                # Создаем временный DataFrame с диапазоном дат и объединяем его с текущим фрагментом данных
                temp_df = pd.merge(dates_df, ts, on='date', how='left')
                temp_df = pd.merge(temp_df, subset, on='date', how='left')
                temp_df['total_sales_in_units'] = temp_df['total_sales_in_units_y'].fillna(temp_df['total_sales_in_units_x'])
                temp_df.rename(columns={'pr_sku_id_x':'pr_sku_id'}, inplace=True)
                temp_df.drop(['total_sales_in_units_x', 'total_sales_in_units_y', 'pr_sku_id_y'], axis=1, inplace=True)
                
                # Заполняем пропущенные значения категориальных столбцов одинаковыми значениями
                temp_df[['st_id', 'pr_sku_id', 'st_city_id', 'st_division_code', 'st_type_format_id', 'st_type_loc_id', \
                        'st_type_size_id', 'pr_group_id', 'pr_cat_id', 'pr_subcat_id', 'pr_uom_id']] = \
                    filling_row[['st_id', 'pr_sku_id', 'st_city_id', 'st_division_code', 'st_type_format_id', 'st_type_loc_id', \
                            'st_type_size_id', 'pr_group_id', 'pr_cat_id', 'pr_subcat_id', 'pr_uom_id']]
                
                # Заполняем уникальный sku_id
                temp_df['unique_sku_id'] = unique_sku_id
                unique_sku_id += 1
                
                # Заполняем пропущенные значения holiday на основе списка праздников
                temp_df['holiday'] = temp_df['date'].isin(self.holidays['date']).astype('category')

                # Установить столбец 'date' как индекс датафрейма
                temp_df.set_index('date', inplace=True)

                # Рассчитываем среднее значение price за последние 14 дней
                temp_df['price_mean'] = temp_df['price'].rolling('14D').mean()

                # Сбросить индекс обратно в столбец
                temp_df.reset_index(inplace=True)

                # Заполняем пропущенные значения нулями
                temp_df['total_sales_in_units'] = temp_df['total_sales_in_units'].fillna(0)
                temp_df['price'] = temp_df['price'].fillna(0)
                temp_df['price_mean'] = temp_df['price_mean'].fillna(0)
                
                # Добавляем заполненные данные в список
                filled_data.append(temp_df)
                
        # Объединяем все заполненные данные в один DataFrame
        filled_df = pd.concat(filled_data)

        # Сброс индекса
        filled_df.reset_index(drop=True, inplace=True)

        return filled_df

    def means_count(self) -> pd.DataFrame:
        '''Метод для сохранения средних цен для использования с прогнозными значениями.'''
        # Загрузка датафрейма от первичных обработчиков
        df = self.fill_missing()
        self.df = df.copy()

        df['mean_all'] = df.groupby('unique_sku_id')['price'].transform('mean')
        # Установить столбец 'date' как индекс датафрейма
        df.set_index('date', inplace=True)

        # Рассчитываем скользящее среднее по временному ряду для каждого уникального значения в столбце 'unique_sku_id'
        df['mean_last'] = df.groupby('unique_sku_id')['price'].rolling('14D').mean().reset_index(level=0, drop=True)

        # Сбросить индекс обратно в столбец
        df.reset_index(inplace=True)

        df['mean_last'] = df['mean_last'].fillna(df['mean_all'])

        # Создаем новый датафрейм means с уникальными айди товаров и соответствующими значениями mean_all и mean_last
        self.means = df.drop_duplicates('unique_sku_id', keep='last')[['unique_sku_id', 'mean_all', 'mean_last']]

        return self.df
    
    def feature_enginering(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Функция создаёт новые признаки для моделей'''
        # Создание признаков с помощью Prophet
        features = []
        future_features = []
        for sku in tqdm(df['unique_sku_id'].unique()):
            # Получаем текущий фрагмент данных для конкретного магазина и товара
            temp = df[(df['unique_sku_id'] == sku)]
            model = Prophet()
            
            # Отключаем вывод дополнительной информации
            model.verbose = False

            model.fit(temp)
            future = model.make_future_dataframe(periods=14, freq='D')
            forecast = model.predict(future)
            features.append(pd.merge(temp, forecast[:-14], on='ds', how='left'))

            forecast['unique_sku_id'] = sku
            future_features.append(forecast[-14:])

        # Объединяем все заполненные данные в один DataFrame
        prophet_df = pd.concat(features)
        prophet_future = pd.concat(future_features)
        # Сброс индекса
        prophet_df.reset_index(drop=True, inplace=True)
        prophet_future.reset_index(drop=True, inplace=True)
        
        file_path = os.path.join(DATA_PATH, f"preprocessed/prophet_future.csv")
        prophet_future.to_csv(file_path, index=False)
        return prophet_df

    def train_preprocess(self) -> pd.DataFrame:
        '''Функция выполняет предобработку данных перед обучением моделей'''
        # Добавим экзогенные признаки
        df_extra = make_date_features(self.means_count().copy())
        df_extra.reset_index(drop=True)

        df_extra['ds'] = pd.to_datetime(df_extra['date'])
        df_extra['y'] = df_extra['total_sales_in_units']
        df_extra.drop(['date', 'total_sales_in_units'], axis=1, inplace=True)

        # Создание новых фичей
        df_extra = self.feature_enginering(df_extra)

        # Выберем экзогенные признаки, категориальные
        CAT_FEATURES = list(df_extra.select_dtypes(include=["category", "object"]).columns)
        df_extra = types_changing(df_extra, CAT_FEATURES)
        EXOG_VARS = ['day', 'month', 'week_number']
        CAT_FEATURES = list(df_extra.select_dtypes(include=["category", "object"]).columns)

        for feature in CAT_FEATURES:
            df_extra[feature] = df_extra[feature].astype('category')

        return df_extra, self.means, CAT_FEATURES, EXOG_VARS
    
    def future_preprocess(self) -> pd.DataFrame:
        '''Функция создания датафрейма для предсказаний'''
        ''' 
        Если были измененены признаки - добавлены или удалены и т.п., то нужно удалить файл future_df.csv, чтобы алгоритм создал его заново.
        Также если вы добавите какие-нибудь признаки, зависящие от предыдущих данных, то нужно в этот алгоритм добавить их генерацию на 
        будущих данных.
        '''
        predict_data = self.df
        means = self.means
        future = pd.DataFrame(columns=predict_data.columns)

        # Определение дат для начала и конца предсказываемого периода
        future_start = predict_data['ds'].max() + timedelta(days=1)
        future_end = future_start + timedelta(days=test_days - 1)
        future_dates = pd.date_range(start=future_start, end=future_end, freq='D')

        for sku in tqdm(predict_data['unique_sku_id'].unique()):
            new_row = predict_data[(predict_data['unique_sku_id'] == sku)].iloc[0]

            # Присвоить значения 'mean_all' и 'mean_last' из словаря Series 'new_row'
            new_row['price'] = means.loc[means['unique_sku_id'] == new_row['unique_sku_id'] \
                                                , 'mean_all'].values[0].astype(float)
            new_row['price_mean'] = means.loc[means['unique_sku_id'] == new_row['unique_sku_id'], \
                                                    'mean_last'].values[0].astype(float)

            for date in future_dates:
                new_row['ds'] = pd.to_datetime(date)
                future.loc[len(future.index)] = new_row

        # Добавляем признак праздничного дня
        future['holiday'] = future['ds'].isin(self.holidays['date']).astype('category')

        return future

    def future_prophet(self) -> pd.DataFrame:    
        future = self.future_preprocess()
        # Добавляем признаки Prophet
        file_path = os.path.join(DATA_PATH, f"preprocessed/prophet_future.csv")
        prophet_future = pd.read_csv(file_path, parse_dates=['ds'])
        cols = prophet_future.columns.to_list()
        cols.remove('ds')
        cols.remove('unique_sku_id')
        future.drop(cols, axis=1, inplace=True)
        
        # Присоединение признаков
        res = []
        for sku in future['unique_sku_id'].unique():
            temp = future[future['unique_sku_id'] == sku]
            temp_prophet = prophet_future[prophet_future['unique_sku_id'] == sku].drop('unique_sku_id', axis=1)
            one_res = pd.merge(temp, temp_prophet, on='ds', how='left')
            res.append(one_res)
        future = pd.concat(res)

        # Удалить целевую переменную
        future.drop('y', axis=1, inplace=True)

        return future


class Optimizer:
    '''Данный класс предназначен для подбора гиперпараметров моделей регрессоров
    Catboost, XGBoost, LightGBM'''
    def __init__(self, df: pd.DataFrame, model: any, cat_features: list, means: pd.DataFrame) -> None:
        self.df = df
        self.train = pd.DataFrame()
        self.future = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.model = model
        self.means = means
        self.CAT_FEATURES = cat_features

    def splitter(self, test_start: datetime, fold: int) -> None:
        df = self.df
        # Задание колонки с датами
        date_column = 'ds'
        
        # Получаем тренировочную выборку
        self.train = df.loc[df[date_column] <= test_start].copy()
        
        # Сохраняем тестовые данные для проверки метрики
        self.test_data = df.loc[df[date_column] > test_start].copy()

        # Создание future датафрейма для тестов
        try:
            file_path = os.path.join(DATA_PATH, f"optuna/{fold}_future_df.csv")
            self.future = pd.read_csv(file_path, parse_dates=['ds'])
            types_changing(self.future, self.CAT_FEATURES)
        except IOError:
            file_path = os.path.join(DATA_PATH, f"optuna/{fold}_future_df.csv")
            preprocessor = Preprocessor(self.train, self.means)
            self.future = preprocessor.future_preprocess()
            # Сохранить в файл
            self.future.to_csv(file_path, index=False)
            # Преобразование данных к правильному типу
            self.future = pd.read_csv(file_path, parse_dates=['ds'])
            types_changing(self.future, self.CAT_FEATURES)

    def tuning_objective(self, trial) -> any:
        '''Функция осуществляет подбор гиперпараметров для модели'''
        
        # Параметры для настройки
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_uniform('subsample', 0.1, 1.0)
        colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.1, 1.0)
        
        if self.model.__class__.__name__ == 'CatBoostRegressor':
            models = [self.model.set_params(random_state=RANDOM_STATE, n_estimators=500, learning_rate=learning_rate, 
                                                    max_depth=max_depth, subsample=subsample,
                                                    colsample_bylevel=colsample_bytree, cat_features=self.CAT_FEATURES)]
        elif self.model.__class__.__name__ == 'XGBRegressor':
            models = [self.model.set_params(random_state=RANDOM_STATE, n_estimators=500, learning_rate=learning_rate, max_depth=max_depth,
                           min_child_weight=min_child_weight, subsample=subsample, colsample_bytree=colsample_bytree, 
                           enable_categorical=True, tree_method='hist')]
        elif self.model.__class__.__name__ == 'LGBMRegressor':
            models = [self.model.set_params(random_state=RANDOM_STATE, n_estimators=500, learning_rate=learning_rate, 
                                                max_depth=max_depth, subsample=subsample,
                                                colsample_bytree=colsample_bytree)]

        lags = trial.suggest_int('lags', 14, 56, step=7) # step means we only try multiples of 7 starting from 14

        fold = n_folds
        scores = []
        while fold > 0:
            # Определяем дату старта тестовой выборки
            test_start = self.df['ds'].max() - timedelta(days=test_days * fold)
            self.splitter(test_start, fold)
 
            forecaster = MLForecast(models=models,
                            freq='D',
                            lags=[1,7, lags],
                            #date_features=['dayofweek', 'month'],
                            num_threads=6)

            forecaster.fit(self.train, id_col='unique_sku_id', time_col='ds', target_col='y', static_features=[])

            p = forecaster.predict(horizon=14, X_df=self.future)
            p = p.merge(self.test_data[['pr_sku_id', 'unique_sku_id', 'ds', 'y']], on=['unique_sku_id', 'ds'], how='left')

            error = wape(p['y'], p[self.model.__class__.__name__])
            scores.append(error)

            fold -= 1

        # Средняя ошибка по фолдам
        print(f'Trial results: {scores}')
        result = np.mean(scores)

        return result

    def tuning_optimize(self, trials: int=20) -> Tuple[float, dict]:
        study = optuna.create_study(direction='minimize')
        study.optimize(self.tuning_objective, n_trials=trials)
        return study.best_value, study.best_params
    
class Predictor:
    '''Класс осуществляет предсказания на основе всего датафрейма в будущее.'''
    def __init__(self, df: pd.DataFrame, future: pd.DataFrame, model: any) -> None:
        self.df = df
        self.future = future
        self.model = model
    
    def make_predictions(self) -> pd.DataFrame:

        df_extra, future_extra = self.df, self.future
        
        models = [self.model]
        
        forecaster = MLForecast(models=models,
                    freq='D',
                    lags=[49],
                    lag_transforms={
                        1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
                    },
                    #date_features=['dayofweek', 'month'],
                    num_threads=6)

        forecaster.fit(df_extra, id_col='unique_sku_id', time_col='ds', target_col='y', static_features=[])

        p = forecaster.predict(horizon=14, X_df=future_extra)
        p = p.merge(future_extra[['st_id', 'pr_sku_id', 'ds', 'unique_sku_id']], on=['unique_sku_id', 'ds'], how='left')
        
        return p