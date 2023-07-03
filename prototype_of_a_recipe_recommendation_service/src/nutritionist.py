from recipes import Forecast, NutritionFacts, SimilarRecipes
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

def main():
    list_of_possible_ingredients = joblib.load('./data/list_of_possible_ingredients.pkl')

    list_of_ingredients = input('Введите список ингридиентов разделяя их запятой с пробелом. Или введите ingredients для вывода возможных ингридиентов').lower()

    if list_of_ingredients == 'ingredients':
        print(list(list_of_possible_ingredients))
        list_of_ingredients = input('Введите список ингридиентов').lower()

    list_of_ingredients = list_of_ingredients.split(', ')
    for elem in list_of_ingredients:
        if elem not in list_of_possible_ingredients:
            print("Один или несколько ингриентов не входит в список возможных ингридиентов, попробуйте снова.")
            return

    n = input("Введите число топ-нутриентов, которые необходимо вывести")
    if  not n.isdigit():
        print("Вы не ввели число, попробуйте еще раз")
        return

    fc = Forecast(list_of_ingredients)
    print(fc.predict_rating_category())

    nf = NutritionFacts(list_of_ingredients, n)
    print(nf.filter())

    sr = SimilarRecipes(list_of_ingredients, 3)
    print(sr.top_similar())

main()