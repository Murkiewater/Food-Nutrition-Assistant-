import sys
from sysconfig import get_python_version
from recipe import get_base_ingredients, get_bin_vector
from config import MAIN_MODEL_PATH
import joblib

def get_nutritions(ingredients, dataframe):
    for ingredient in ingredients:
        if ingredient in dataframe.index:
            print(ingredient)
            for col, value in dataframe.loc[ingredient].dropna().items():
                if pd.isna(value):
                    continue
                print(f'{col} - {value}% of Daily Value')
            print()
        else:
            print(f'{ingredient} is not in nutrition facts dataframe!')
            return 0
    return 1

def main():
    if len(sys.argv) == 1:
        print("There must be several ingredients")
        sys.exit(1)

    # Читаем список ингредиентов
    ingredients = " ".join(sys.argv[1:])
    ingredients = [word.strip().lower() for word in ingredients.split(',') if word.strip()]

    # Инициализируем список базовых ингдериентов
    base_ingredients = get_base_ingredients()

    # Инициализируем список базовых ингдериентов
    # base_ingredients = get_base_ingredients()
    

    # Получаем вектор для предсказания модели
    bin_vector = get_bin_vector(ingredients, base_ingredients)
    # Предсказываем значение по вектору
    model = joblib.load(MAIN_MODEL_PATH)
    score = model.predict(bin_vector)

    # Выводим прогноз в зависимости от предсказания
    print(score[0])
    

    # Выводим список нутриентов для каждого ингредиента
    nutrition_df = pd.read_csv('src/data/nutrition_facts.csv', index_col='ingredient')
    if (get_nutritions(ingredients, nutrition_df) == 0):
        sys.exit(1)

    

    # Добавляем сколько это от суточной нормы

    # Сортируем исходный dataset по приближенному расстоянию
    # Находим ближайшее косинусное расстояние
    # Есть ли у него url выводим 3 значения.


if __name__ == "__main__":
    main()
