import pandas as pd
import numpy as np
import re
import datetime
import spacy
from joblib import load

def remove_symbols(doc):
    word_pattern = re.compile("^[а-яё]*$")
    return [token for token in doc if word_pattern.match(token)]

def preprocess_features(row):
    # Разделим значения столбца floor на два признака
    row['total_floors'] = row['floor'].apply(lambda x: x.split('/')[1])
    row['floor'] = row['floor'].apply(lambda x: x.split('/')[0])

    # Приведем тип данных к int
    row['total_floors'] = row['total_floors'].astype('int64')
    row['floor'] = row['floor'].astype('int64')

    # В столбцах с date_app и date_update оставим только дату, без времени и переведем в соответствующий формат
    row['date_app'] = pd.to_datetime(row['date_app'].apply(lambda x: x.split()[0]), format='%d.%m.%Y')
    row['date_update'] = pd.to_datetime(row['date_update'].apply(lambda x: x.split()[0]), format='%d.%m.%Y')

    # Приведем все строковые значения к нижнему регистру
    cat_columns = ['url', 'note', 'apartment_type', 'neighborhood', 'street', 'layout']
    row[cat_columns] = row[cat_columns].apply(lambda row: row.str.lower(), axis=1)

    # Убираем цифры и лишние символы в названиях улиц
    row['street'] = row['street'].apply(lambda x: x.replace('ул ', ''))
    row['street'] = row['street'].apply(lambda x: x.replace('ул. ', ''))
    row['street'] = row['street'].apply(lambda x: x.replace(' пр-т', ''))
    row['street'] = row['street'].apply(lambda x: re.sub(r'[^а-яё\s]', '', x).strip())

    # Удаляем столбцы layout и url, house
    row = row.drop('layout', axis=1)
    row = row.drop('url', axis=1)
    row = row.drop('house', axis=1)

    # Новые признаки
    row['age'] = round((datetime.datetime.today() - row['date_app']) / np.timedelta64(1, 'D'), 2)
    row['living/total'] = round(row['living_area'] / row['total_area'], 2)
    row['kitchen/living'] = round(row['kitchen_area'] / row['living_area'], 2)

    row = row.fillna(0)
    row.loc[row['kitchen/living'] == np.inf, 'kitchen/living'] = 999

    row['mean_views'] = round(row['views'] / row['age'], 2)
    row['len_text'] = row['note'].apply(lambda x: len(x))

    # Определим модель spacy
    model = spacy.load('ru_core_news_sm', disable=['ner', 'parser'])
    texts = row['note'].tolist()
    res = []
    for doc in model.pipe(texts, disable=["tagger", "parser"]):
        res.append([token.lemma_ for token in doc])

    # Очистим текст
    corpus = list(map(remove_symbols, res))

    # Объединим списки слов в предложения
    docs = [" ".join(tokens) for tokens in corpus]

    tfidf = load('../tfidf.pkl')
    # Трансформируем корпус
    x = tfidf.transform(docs)

    # Уменьшаем размерность
    pca = load('../pca.pkl')
    x = pca.transform(x.toarray())

    # Объединяем данные
    row = pd.concat([row, pd.DataFrame(x, columns=['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5'])],
                   axis=1)
    row = row.drop(['note'], axis=1)
    row = row.drop(['date_app'], axis=1)
    row = row.drop(['date_update'], axis=1)

    return row
