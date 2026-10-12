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
