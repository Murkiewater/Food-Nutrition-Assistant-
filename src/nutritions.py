import sys
from recipe import (
    get_base_ingredients,
    get_bin_list,
    get_df_with_urls,
    get_json_urls,
    output_recommendation,
)
from config import (
    MAIN_MODEL_PATH,
    RECIPES_JSON_PATH,
    RECIPES_DB_PATH,
    PROCESSED_DATASET_PATH,
)
import joblib
import pandas as pd

rmse_intervals = [
    ((float("-inf"), 2), "Низкий уровень"),
    ((3, 4), "Средний уровень"),
    ((4, float("inf")), "Высокий уровень"),
]


def get_nutritions(ingredients, dataframe):
    for ingredient in ingredients:
        if ingredient in dataframe.index:
            print(ingredient)
            for col, value in dataframe.loc[ingredient].dropna().items():
                if pd.isna(value):
                    continue
                print(f"{col} - {value}% of Daily Value")
            print()
        else:
            print(f"{ingredient} is not in nutrition facts dataframe!")
            return 0
    return 1


def output_part_1(ingredients):
    # Инициализируем список базовых ингдериентов
    base_ingredients = get_base_ingredients()

    # Получаем вектор для предсказания модели
    bin_vector = get_bin_list(ingredients, base_ingredients)
    # Предсказываем значение по вектору
    model = joblib.load(MAIN_MODEL_PATH)
    score = model.predict(bin_vector)

    # Выводим прогноз в зависимости от предсказания
    print("I. OUR FORECAST")

    status = "Неизвестно"
    for (start, end), label in rmse_intervals:
        if start <= score <= end:
            status = label
            break

    print(status)


def output_part_2(ingredients):
    print("II. NUTRITION FACTS")

    # Выводим список нутриентов для каждого ингредиента
    nutrition_df = pd.read_csv("src/data/nutrition_facts.csv", index_col="ingredient")
    if get_nutritions(ingredients, nutrition_df) == 0:
        sys.exit(1)


def output_part_3(ingredients):
    print("III. TOP-3 SIMILAR RECIPES:")

    df = get_df_with_urls(reading_columns_path=PROCESSED_DATASET_PATH)
    output_recommendation(df, ingredients, base_ingredients_path=PROCESSED_DATASET_PATH)


def main():
    if len(sys.argv) == 1:
        print("There must be several ingredients")
        sys.exit(1)

    # Читаем список ингредиентов
    # ingredients = " ".join(sys.argv[1:])
    ingredients = sys.argv[1:]
    # ingredients = [word.strip().lower() for word in ingredients.split(',') if word.strip()]
    output_part_1(ingredients=ingredients)
    output_part_2(ingredients=ingredients)
    output_part_3(ingredients=ingredients)

    # Добавляем сколько это от суточной нормы

    # Сортируем исходный dataset по приближенному расстоянию
    # Находим ближайшее косинусное расстояние
    # Есть ли у него url выводим 3 значения.


if __name__ == "__main__":
    main()
