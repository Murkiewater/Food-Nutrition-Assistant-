import sys
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from config import PROCESSED_DATASET_PATH


def get_base_ingredients(file_pat=PROCESSED_DATASET_PATH):
    # читаем файл с ингредиентами
    with open(file_pat, "r") as f:
        line = f.readline()
    base_ingredients = [word.strip() for word in line.split(sep=",")]
    return base_ingredients


def get_bin_vector(data, base_data):
    mlb = MultiLabelBinarizer(classes=base_data)
    mlb.fit(base_data)

    return mlb.transform(data)


def main():
    if len(sys.argv) == 1:
        print("There must be several ingredients")

    # Читаем список ингредиентов
    ingredietns = sys.arg[1:]

    # Инициализируем список базовых ингдериентов
    base_ingredients = get_base_ingredients()
    

    # Получаем вектор для предсказания модели
    # Возможно, сделать читаемым из файла

    # Предсказываем значение по вектору

    # Выводим прогноз в зависимости от предсказания

    # Выводим список нутриентов для каждого ингредиента
    # Добавляем сколько это от суточной нормы

    # Сортируем исходный dataset по приближенному расстоянию
    # Находим ближайшее косинусное расстояние
    # Есть ли у него url выводим 3 значения.


if __name__ == "__main__":
    main()
