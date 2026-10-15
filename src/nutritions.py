import sys
from sysconfig import get_python_version
from recipe import get_base_ingredients, get_bin_vector
from config import MAIN_MODEL_PATH
import joblib


def main():
    if len(sys.argv) == 1:
        print("There must be several ingredients")

    # Читаем список ингредиентов
    ingredients = sys.argv[1:]

    # Инициализируем список базовых ингдериентов
    base_ingredients = get_base_ingredients()
    # Получаем вектор для предсказания модели
    bin_vector = get_bin_vector(ingredients, base_ingredients)
    # Предсказываем значение по вектору
    model = joblib.load(MAIN_MODEL_PATH)
    score = model.predict(bin_vector)

    # Выводим прогноз в зависимости от предсказания
    print(score[0])
    

    # Выводим список нутриентов для каждого ингредиента

    # Добавляем сколько это от суточной нормы

    # Сортируем исходный dataset по приближенному расстоянию
    # Находим ближайшее косинусное расстояние
    # Есть ли у него url выводим 3 значения.


if __name__ == "__main__":
    main()
