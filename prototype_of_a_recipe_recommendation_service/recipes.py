import pandas as pd
import joblib
from random import randrange

class Forecast():
    """
    Предсказание рейтинга блюда или его класса
    """
    def __init__(self, list_of_ingredients):
        self.list_of_ingredients = list_of_ingredients
        self.list_of_possible_ingredients = joblib.load('./data/list_of_possible_ingredients.pkl')

    def preprocess(self):
        """
        Этот метод преобразует список ингредиентов в структуры данных,
        которые используются в алгоритмах машинного обучения, чтобы сделать предсказание.
        """
        vector = []

        for elem in self.list_of_possible_ingredients:
            if elem in self.list_of_ingredients:
                vector.append(1)
            else:
                vector.append(0)
        vector = pd.DataFrame([vector], columns=self.list_of_possible_ingredients)
        return vector

    def predict_rating(self):
        """
        Этот метод возвращает рейтинг для списка ингредиентов, используя регрессионную модель,
        которая была обучена заранее. Помимо самого рейтинга, метод также возвращает текст,
        который дает интерпретацию этого рейтинга и дает рекомендацию, как в примере выше.
        """
        model = joblib.load('best_model_reg.pkl')
        rating = model.predict(self.preprocess())

        if rating <=1:
            text = "На это не стоит тратить продукты"
        elif rating <=3:
            text= "Есть можно, если сильно голодный"
        else:
            text="Похоже на вкусняшку!"

        return f'I. НАШ ПРОГНОЗ \n\n{text} \n\n'

    def predict_rating_category(self):
        """
        Этот метод возращает рейтинговую категорию для списка ингредиентов, используя классификационную модель,
        которая была обучена заранее. Помимо самого рейтинга, метод возвращает также и текст,
        который дает интерпретацию этой категории и дает рекомендации, как в примере выше.
        """
        model = joblib.load('best_model_clf.pkl')
        rating = model.predict(self.preprocess())

        if rating == 'bad':
            text = "На это не стоит тратить продукты"
        elif rating == 'so-so':
            text="Есть можно, если сильно голодный"
        else:
            text="Похоже на вкусняшку!"

        return f'I. НАШ ПРОГНОЗ \n\n{text} \n\n'

class NutritionFacts():
    """
    Выдает информацию о пищевой ценности ингредиентов.
    """
    def __init__(self, list_of_ingredients, n):
        self.list_of_ingredients = list_of_ingredients
        self.nutrients = pd.read_csv('./data/nutrients_daily_value.csv')
        self.n = int(n)

    def retrieve(self):
        """
        Этот метод получает всю имеющуюся информацию о пищевой ценности из файла с заранее собранной информацией по заданным ингредиентам.
        Он возвращает ее в том виде, который вам кажется наиболее удобным и подходящим.
        """
        return self.nutrients.loc[self.nutrients['item'].isin(self.list_of_ingredients)].sort_values(by=['item', 'value'], ascending=False)

    def filter(self):
        """
        Этот метод отбирает из всей информации о пищевой ценности только те нутриенты, которые были заданы в must_nutrients (пример в PDF-файле ниже),
        а также топ-n нутриентов с наибольшим значением дневной нормы потребления для заданного ингредиента.
        Он возвращает текст, отформатированный как в примере выше.
        """
        text_with_facts = "II. ПИЩЕВАЯ ЦЕННОСТЬ\n"
        f = self.retrieve()
        for elem in self.list_of_ingredients:
            text_with_facts += f'\n{elem}\n'
            for i in range(self.n):
                nutrient_name = f.loc[f['item']==elem].iloc[i]['nutrient_name']
                value = round(f.loc[f['item']==elem].iloc[i]['value'], 2)
                text_with_facts += f'{nutrient_name} - {value}% of Daily Value\n'
        return text_with_facts

class SimilarRecipes:
    """
    Рекомендация похожих рецептов с дополнительной информацией
    """
    def __init__(self, list_of_ingredients, n):
        self.list_of_ingredients = list_of_ingredients
        self.recipes = pd.read_csv('./data/url_result.csv')
        self.n = n

    def find_all(self):
        """
        Этот метод возвращает список индексов рецептов, которые содержат заданный список ингредиентов.
        Если нет ни одного рецепта, содержащего все эти ингредиенты, то сделайте обработку ошибки, чтобы программа не ломалась.
        """
        indexes = self.recipes.loc[self.recipes[self.list_of_ingredients].sum(axis=1)==len(self.list_of_ingredients)].index
        return indexes

    def top_similar(self):
        """
        Этот метод возвращает текст, форматированный как в примере выше: с заголовком, рейтингом и URL.
        Чтобы это сделать, он вначале находит топ-n наиболее похожих рецептов с точки зрения количества дополнительных ингредиентов,
        которые потребуются в этих рецептах. Наиболее похожим будет тот, в котором не требуется никаких других ингредиентов.
        Далее идет тот, у которого появляется 1 доп. ингредиент. Далее – 2.
        Если рецепт нуждается в более, чем 5 доп. ингредиентах, то такой рецепт не выводится.
        """
        indexes = self.find_all()
        indx = self.recipes.loc[indexes].iloc[:, 3:].sum(axis=1).sort_values()
        real_indx = indx.index[indx<=8]
        text_with_recipes = f"III. ТОП-3 ПОХОЖИХ РЕЦЕПТА:\n"

        if len(real_indx)==0:
            text_with_recipes += f"Похожих рецептов не найдено. \n\n"
        elif len(indexes)>=3:
            for i in range(3):
                p = self.recipes.iloc[real_indx[i]]
                text_with_recipes += f"{p['title']}, рейтинг: {p['rating_url']}, URL: {p['url']} \n"
        else:
            for i in range(len(indexes)):
                p = self.recipes.iloc[real_indx[i]]
                text_with_recipes += f"{p['title']}, рейтинг: {p['rating_url']}, URL: {p['url']} \n"
                text_with_recipes += f"Больше похожих рецептов не найдено. \n\n"
        return text_with_recipes
