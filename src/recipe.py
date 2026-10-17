import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from bs4 import BeautifulSoup as bs
import abc
from urllib.request import urlopen as url
from config import (
    PROCESSED_DATASET_PATH,
    INGREDIENTS,
    RECIPES_DB_PATH,
    RECIPES_JSON_PATH,
)
import pandas as pd
import sqlite3

from tqdm import tqdm
import numpy as np
import re

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    ParameterGrid,
    cross_val_score,
)
from sklearn.metrics import accuracy_score, root_mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin

# fmt: off
# must be passed raw html
class Recipe:
    __metaclass__ = abc.ABCMeta
    title = ''
    date = ''
    desc = ''
    ingredients = []
    directions = []
    categories = []

    @abc.abstractstaticmethod
    def get_title(self, page):
        pass

    @abc.abstractstaticmethod
    def get_ingredients(self, page):
        pass

    @abc.abstractstaticmethod
    def get_directions(self, page):
        pass

    @abc.abstractstaticmethod
    def get_categories(self, page):
        pass

    @abc.abstractstaticmethod
    def get_date(self, page):
        pass

    @abc.abstractstaticmethod
    def get_desc(self, page):
        pass

    def build_recipe(self, page):
        self.title = self.get_title(page)
        self.ingredients = self.get_ingredients(page)
        self.directions = self.get_directions(page)
        self.categories = self.get_categories(page)
        self.date = self.get_date(page)
        self.desc = self.get_desc(page)

    def __init__(self, page):
        print('attempting to build from: '+page)
        try:
            self.build_recipe(bs(url(page), 'html.parser'))
        except Exception as x:
            print('Could not build from %s, %s'%(page,x))

class FN_Recipe(Recipe):
    def get_title(self, page):
        return page.find('div', {'class': 'tier-3 title'}).text.encode('utf-8').strip()

    def get_ingredients(self, page):
        return [i.text.encode('utf-8').strip() for i in
                page.find_all('div', {'class': 'ingredients'})[0].find_all('li')]

    def get_directions(self, page):
        return [i.text.encode('utf-8').strip() for i in
                page.find_all('ul', {'class': 'recipe-directions-list'})[0].find_all('li')]

    def get_categories(self, page):
        return [i.text.encode('utf-8').strip() for i in page.find_all('ul', {'class': 'categories'})[0].find_all('li')]

    def get_date(self, page):
        try:
            page_s = str(page)
            return page_s[page_s.index('OrigPubDate') + 14:page_s.index('OrigPubDate') + 24]
        except:
            return ''

    def get_desc(self, page):
        return [dd['content'] for dd in page.find_all('meta', {'itemprop': 'description'})][-1].encode('utf-8').strip()

# No scripting shit apparently needed
class EP_Recipe(Recipe):
    rating = None
    calories = None
    sodium = None
    fat = None
    protein = None


    def get_date(self, page):
        try:
            return page.find('meta', {'itemprop': 'datePublished'})['content']
        except:
            return None

    def get_desc(self, page):
        try:
            return page.find('div', {'itemprop': 'description'}).find('p').text
        except:
            return None

    def get_directions(self, page):
        return [i.text.strip() for i in page.find_all('li', {'class': "preparation-step"})]

    def get_ingredients(self, page):
        return [i.text.strip() for i in page.find_all('li', {'itemprop': "ingredients"})]

    def get_categories(self, page):
        return [i.text for i in page.find_all('dt', {'itemprop': "recipeCategory"})]

    def get_title(self, page):
        return page.find('h1', {'itemprop': 'name'}).text

    def get_rating(self, page):
        try:
            return float(page.find_all('span', {'class': 'rating'})[-1].text.split('/')[0]) * 5 / 4
        except:
            return None

    def build_recipe(self, page):
        super(EP_Recipe, self).build_recipe(page)
        self.rating = self.get_rating(page)
        self.calories = self.get_calories(page)
        self.sodium = self.get_sodium(page)
        self.fat = self.get_fat(page)
        self.protein = self.get_protein(page)

    def get_calories(self,page):
        try:
            return float(page.find('span',{'class':'nutri-data','itemprop':'calories'}).text)
        except:
            return None

    def get_sodium(self,page):
        try:
            return float(page.find('span',{'class':'nutri-data','itemprop':'sodiumContent'}).text.split(' ')[0])
        except:
            return None

    def get_fat(self, page):
        try:
            return float(page.find('span', {'class': 'nutri-data', 'itemprop': 'fatContent'}).text.split(' ')[0])
        except:
            return None

    def get_protein(self, page):
        try:
            return float(page.find('span', {'class': 'nutri-data', 'itemprop': 'proteinContent'}).text.split(' ')[0])
        except:
            return None
# fmt: on


def get_bin_list(data, base_data):
    input_set = set(data)
    bin_vector = [1 if word in input_set else 0 for word in base_data]

    return pd.DataFrame(data=[bin_vector], columns=base_data)


def get_base_ingredients(file_pat=PROCESSED_DATASET_PATH):
    # читаем файл с ингредиентами
    all_data = pd.read_csv(file_pat, nrows=0).columns.tolist()
    ing_selector = StrictIngredientSelector()
    base_ingredients = ing_selector.filter_columns(all_data)
    return base_ingredients


def prepare_json_df(df):
    df = df.dropna(how="all")
    df_cols = ["fat", "calories", "protein", "rating", "sodium", "title", "desc"]
    df["index_row"] = df.index
    df = df.drop_duplicates(subset=df_cols)
    return df


def get_json_urls(urls_db=RECIPES_DB_PATH, json_path=RECIPES_JSON_PATH):
    # импорт json данных
    query = "SELECT * FROM parsed_recipes"
    conn = sqlite3.connect(urls_db)
    parsed_urls = pd.io.sql.read_sql(query, conn)
    df_json = pd.read_json(json_path)
    df_json = prepare_json_df(df_json)
    # соединить с пропасенными данными
    rename_cols = {"date": "recipe_date"}
    json_urls = (
        pd.merge(
            left=df_json,
            right=parsed_urls,
            left_on=["title", "index_row"],
            right_on=["title", "df_index"],
        )
        .drop(columns=["df_index", "index_row"])
        .rename(columns=rename_cols)
    )
    return json_urls


def get_df_with_urls(reading_columns_path="data/epi_r_processed.csv"):
    # соединить с исходными данными
    json_urls = get_json_urls()
    epi_rp = pd.read_csv(reading_columns_path)
    df = pd.merge(
        left=epi_rp,
        right=json_urls,
        on=["title", "protein", "calories", "fat", "sodium", "rating"],
    )
    mask_empty_urls = df["recipe_url"] == ""
    df = df[~mask_empty_urls]
    return df


def get_full_epicurious_url(url):
    return "https://epicurious.com" + url


class StrictIngredientSelector:
    def __init__(self):
        # 1. Список чистых кулинарных корней (только существительные-продукты)
        # fmt: off
        food_words = [
            'almond', 'amaretto', 'anchovy', 'anise', 'apple', 'apricot', 'artichoke', 'arugula', 'pear', 'asparagus',
            'avocado', 'bacon', 'banana', 'barley', 'basil', 'bean', 'beef', 'beer', 'beet', 'berry', 'biscuit', 
            'bitters', 'blackberry', 'blueberry', 'bok choy', 'bourbon', 'bran', 'brandy', 'bread', 'breadcrumbs', 
            'brie', 'brisket', 'broccoli', 'brussel sprout', 'butter', 'buttermilk', 'squash', 'cabbage', 'cake', 
            'calvados', 'campari', 'candy', 'cantaloupe', 'capers', 'caraway', 'cardamom', 'carrot', 'cashew', 
            'cauliflower', 'caviar', 'celery', 'champagne', 'chard', 'chartreuse', 'cheddar', 'cheese', 'cherry', 
            'chestnut', 'chicken', 'chickpea', 'chile', 'chili', 'chocolate', 'cinnamon', 'citrus', 'clam', 'clove', 
            'coconut', 'cod', 'coffee', 'coriander', 'corn', 'cornmeal', 'couscous', 'crab', 'cranberry', 'cream', 
            'cucumber', 'cumin', 'currant', 'curry', 'custard', 'dairy', 'date', 'dill', 'duck', 
            'egg', 'eggplant', 'endive', 'escarole', 'fennel', 'feta', 'fig', 'fish', 'fontina', 'frangelico', 'fruit', 
            'garlic', 'ginger', 'gin', 'goat cheese', 'goose', 'gouda', 'grains', 'grape', 'grapefruit', 
            'grappa', 'green bean', 'onion', 'scallion', 'guava', 'halibut', 'ham', 'hazelnut', 
            'honey', 'honeydew', 'horseradish', 'hummus', 'ice cream', 'jalapeño', 'jam', 'jelly', 'jícama', 'kahlúa', 
            'kale', 'kiwi', 'kumquat', 'lamb', 'leek', 'legume', 'lemon', 'lemongrass', 'lentil', 'lettuce', 
            'lime', 'lingonberry', 'liqueur', 'lobster', 'lychee', 'macadamia', 'mango', 'maple syrup', 'margarita', 
            'marsala', 'marscarpone', 'marshmallow', 'martini', 'mayonnaise', 'meat', 'melon', 'mezcal', 'milk', 
            'mint', 'molasses', 'monterey jack', 'mozzarella', 'muffin', 'mushroom', 'mussel', 'mustard', 'nectarine', 
            'noodle', 'nut', 'nutmeg', 'oat', 'oatmeal', 'octopus', 'okra', 'olive', 'orange', 'oregano', 
            'orzo', 'oyster', 'pancake', 'papaya', 'paprika', 'parmesan', 'parsley', 'parsnip', 'pasta', 'pastry', 
            'pea', 'peach', 'peanut', 'pecan', 'pepper', 'pernod', 'persimmon', 'pickles', 'pine nut', 
            'pineapple', 'pistachio', 'plantain', 'plum', 'poblano', 'pomegranate', 'poppy', 'pork', 
            'port', 'potato', 'poultry', 'prosciutto', 'prune', 'pumpkin', 'punch', 'quail', 'quince', 
            'quinoa', 'rabbit', 'radicchio', 'radish', 'raisin', 'ramekin', 'raspberry', 'red wine', 'rhubarb', 
            'rice', 'ricotta', 'rosemary', 'rosé', 'rum', 'rutabaga', 'rye', 'saffron', 'sage', 'sake', 'salad', 
            'salmon', 'salsa', 'sangria', 'sardine', 'sausage', 'scallop', 'scotch', 
            'semolina', 'sesame', 'shallot', 'shellfish', 'sherry', 'shrimp', 'snapper', 'soufflé', 
            'stew', 'sour cream', 'soy', 'soy sauce', 'sparkling wine', 'spice', 'spinach', 
            'squid', 'steak', 'stock', 'strawberry', 'sweet potato', 'swiss cheese', 'taco', 
            'tamarind', 'tangerine', 'tapioca', 'tart', 'tea', 'tequila', 'thyme', 'tilapia', 'tofu', 'tomatillo', 
            'tomato', 'tortillas', 'triple sec', 'trout', 'tuna', 'turkey', 'turnip', 'vanilla', 'veal', 
            'venison', 'vermouth', 'vinegar', 'vodka', 'walnut', 'wasabi', 'watercress', 'watermelon', 
            'white wine', 'whiskey', 'wild rice', 'wine', 'yuca', 'zucchini', 
            'créme de cacao', 'yogurt', 'bulgur', 'kirsch', 'swordfish', 'chambord', 'collard greensa'
        ]
        # fmt: on

        # Компилируем регулярное выражение строго по границам целых слов (\b)
        self.food_pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in food_words) + r")\b",
            flags=re.IGNORECASE,
        )

        # Паттерны стоп-слов, которые превращают еду в мета-теги (например, "peanut free")
        self.meta_blacklist = re.compile(
            r"#|\d|free|low|less|added|conscious|friendly|meals|recipes|prep|required|maker",
            flags=re.IGNORECASE,
        )

    def filter_columns(self, columns_list):
        cleaned_ingredients = []
        for col in columns_list:
            col_lower = str(col).lower().strip()

            # 1. Если строка содержит технический мусор или маркеры диет — жестко пропуeскаем
            if self.meta_blacklist.search(col_lower):
                continue

            # 2. Проверяем строгое совпадение с базой продуктов по границам слов
            if self.food_pattern.search(col_lower):
                cleaned_ingredients.append(col)

        return cleaned_ingredients


class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, save_to_csv=False, output_path="data/epi_r_processed.csv"):
        super().__init__()
        self.save_to_csv = save_to_csv
        self.output_path = output_path

    def fit(self, X):
        return self

    def transform(self, X):
        # check columns
        if hasattr(X, "columns"):
            # удаление дубликатов в исходном df
            X_cols = ["title", "rating", "protein", "calories", "fat", "sodium"]
            print(f"[DataCleaner] After drop duplicates. Shape = {X.shape}")
            X = X.drop_duplicates(subset=X_cols)
            print(f"[DataCleaner] After drop duplicates. Shape = {X.shape}")

            mask_cols_eq2 = X.nunique() == 2
            eq2_cols = list(X.columns[mask_cols_eq2])

            print(f"[DataCleaner] Before memory optimization")
            print(X.info(memory_usage=True))
            print()

            for column in X.columns[mask_cols_eq2]:
                X[column] = pd.to_numeric(X[column], downcast="integer")

            print(f"[DataCleaner] After memory optimization")
            print(X.info(memory_usage=True))
            print()

            selector = StrictIngredientSelector()
            final_ingredients = selector.filter_columns(eq2_cols)

            print(f"[DataCleaner] Final ingredients count: {len(final_ingredients)}")
            X_new = X[X_cols + final_ingredients].copy()
            if self.save_to_csv:
                X_new.to_csv(self.output_path, index=False)
                print(
                    f"[DataCleaner] Набор данных сохранен ({X_new.shape}) сохранен в csv {self.output_path}"
                )

            return X_new
        else:
            raise Exception("[DataCleaner] There is no such column attribute.")


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.target_name = "rating"

    def fit(self, X):
        return self

    def transform(self, X):
        drop_cols = ["fat", "calories", "protein", "sodium", "title"]
        if self.target_name in X.columns:
            X = X.drop(columns=drop_cols)
            print(f"[FeatureExtractor] Feature extracted. {X.shape}")
            return X
        else:
            raise Exception(f"[FeatureExtractor] There is no col ratings in dataset.")


class TrainValidationTest:

    def __init__(self, test_size=0.2, random_state=21):
        self.test_size = test_size
        self.random_state = random_state

    def get_datasets(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_train,
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test


class ModelSelectionRegression:
    def __init__(self, grids: GridSearchCV):
        self.grids = grids
        self.trained = False
        self.model_results = {}

    def choose(self, X_train, y_train, X_valid, y_valid):
        # process of training
        for i, grid in enumerate(self.grids):

            param_grid = grid.param_grid
            estimator = grid.estimator
            cv = grid.cv
            name = estimator.__class__.__name__

            best_score = float("inf")
            # best_estimator = None
            best_params = None

            print(f"Estimator: {name}")
            for params in tqdm(ParameterGrid(param_grid)):
                estimator.set_params(**params)
                score = cross_val_score(
                    estimator,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                )
                mean_score = -np.mean(score)

                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params

            row_results = {
                "name": grid,
                "params": best_params,
                "valid_score": best_score,
            }
            print(f"Best params: {best_params}")
            print(f"Best training RMSE: {best_score:.3f}")

            y_pred = (
                estimator.set_params(**params).fit(X_train, y_train).predict(X_valid)
            )
            valid_score = root_mean_squared_error(y_valid, y_pred)
            print(f"Validation set RMSE score for best params: {valid_score:.3}")
            print()

            self.model_results[i] = row_results

        best_model_name = (
            pd.DataFrame(self.model_results)
            .T.sort_values(by="valid_score", ascending=True)
            .iloc[0]["name"]
        )
        print(f"Classifier with best validation set RMSE: {best_model_name}")

        return best_model_name


def output_cross_val_results(cvs_result, model_name=""):
    print(f"""Cross-validation sore of mode {model_name}
mean = {-np.mean(cvs_result):.2f}, std={np.std(cvs_result):.2f}""")


class Tests:

    def test_get_base_ingredients(self):
        result = get_base_ingredients()
        expected = INGREDIENTS
        assert result == expected


def output_recommendation(df, words, base_ingredients_path="data/epi_r_processed.csv"):
    # определяем вектор
    base_data = get_base_ingredients(file_pat=base_ingredients_path)
    bin_list = get_bin_list(words, base_data=base_data)
    # для каждого слова считаем косинусное расстояние и сохраняем вdf
    bin_array = np.array(bin_list)
    # матрица
    m = df[base_data].to_numpy()
    # считаем метрику Жаккара = Пересечение / Объединение
    intersection = np.logical_and(m, bin_array).sum(axis=1)
    union = np.logical_or(m, bin_array).sum(axis=1)
    jaccard_scores = np.zeros_like(intersection, dtype=float)

    # Делим ТОЛЬКО там, где объединение больше нуля (избегаем деления на 0)
    mask = union > 0
    jaccard_scores[mask] = intersection[mask] / union[mask]

    jaccard_series = pd.Series(jaccard_scores, index=df.index).sort_values(
        ascending=False
    )
    jaccard_series.head(20)
    # сортируем по значению косинусного возвращаем столбец с URL (удаляем пропуски)
    final_series = jaccard_series
    url_ind_3 = final_series[:10].index
    # print(url_ind_3)
    for ind, row in df.loc[url_ind_3, :].iterrows():
        url = get_full_epicurious_url(row["recipe_url"])
        print(f" - {row['title'].strip()}, rating: {row['rating']}, URL: {url}")


# if __name__ == "__main__":
#     import pickle
#     import json
#     import multiprocessing

#     # ep_urls = pickle.load(open('epi_urls.checkpoint','rb'))
#     # ep_urls = [str(i) for i in ep_urls]
#     # p = multiprocessing.Pool(4)
#     # output = p.map(EP_Recipe,ep_urls)
#     # pickle.dump(output,open('epi_recipes.final','wb'))
#     data = pickle.load(open("epi_recipes.final", "rb"))
#     ar = []
#     for i in data:
#         ar.append(i.__dict__)
#     pickle.dump(ar, open("epi_recipe_dict_form.dict", "wb"))

#     with open("result.json", "w") as fp:
#         json.dump(ar, fp)
