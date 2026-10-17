import sys
from recipe import get_base_ingredients, get_bin_list
from config import MAIN_MODEL_PATH
import joblib
import numpy as np

def main():
    if len(sys.argv) == 1:
        print("There must be several ingredients")

    # Читаем список ингредиентов
    ingredients = sys.argv[1:]

    # Инициализируем список базовых ингдериентов
    base_ingredients = get_base_ingredients()
    # Получаем вектор для предсказания модели
    bin_vector = get_bin_list(ingredients, base_ingredients)
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
def output_recomendation(df, words, base_ingredients_path='data/epi_r_processed.csv', recipe_num=3):
    # определяем вектор
    base_data = get_base_ingredients(file_pat=base_ingredients_path)
    binv= get_bin_list(words, base_data=base_data)
    # для каждого слова считаем косинусное расстояние и сохраняем вdf
    binv_array = np.array(binv)
    # матрица
    m = df[base_data].to_numpy()
    # считаем метрику Жаккара = Пересечение  / Объединение
    intersection = np.logical_and(m, binv_array).sum(axis=1)
    union = np.logical_or(m, binv_array).sum(axis=1)
    jaccard_scores = np.zeros_like(intersection, dtype=float)

    # Делим ТОЛЬКО там, где объединение больше нуля (избегаем деления на 0)
    mask = union > 0
    jaccard_scores[mask] = intersection[mask] / union[mask]

    # сортируем по значению косинусного возвращаем столбец с URL 
    jaccard_series = pd.Series(jaccard_scores, index=df.index).sort_values(ascending=False)
    final_series = jaccard_series
    final_idx = final_series[:recipe_num].index
    for ind, row in df.loc[final_idx, :].iterrows():
        url = get_full_epicurious_url(row['recipe_url'])
        print(f' - {row['title'].strip()}, rating: {row['rating']}, URL: {url}')    

if __name__ == "__main__":
    main()
