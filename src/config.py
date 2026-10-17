from pathlib import Path
import logging

PARSING_NUMBER_ATTEMPTS = 3
RECIPES_JSON_PATH = Path("src/data/full_format_recipes.json")
RECIPES_DB_PATH = Path("src/data/recipe_urls.db")
PARSING_URL_LOG_FILE = Path("src/parse_url.log")
PROCESSED_DATASET_PATH = Path("src/data/epi_r.csv")
MAIN_MODEL_PATH = Path("src/data/models/rf_clf_rmse_1.2713903560555384.sav")


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://google.com",
}


INIT_DB = False

SEARCH_URL = "https://epicurious.com/search?q={title}".format
LOGGING_LEVEL = logging.INFO
TAGS_INFO = {
    "title": {
        "parent": {"name": "div", "class_": "content-header-text"},
        "child": {
            "name": "h1",
            "class_": "BaseText-fEwdHD SplitScreenContentHeaderHed-fzFWyi hTEWmH sxfAs",
        },
    },
    "ingredients": {
        "parent": {
            "name": "div",
            "class_": "Wrapper-duGgyO cZuvuY",
        },
        "child": {
            "name": "div",
            "class_": "BaseText-fEwdHD Description-dTzQRt dcnHdL egeGnB",
        },
    },
    "directions": {
        "parent": {"name": "div", "class_": "InstructionsWrapper-gtQceH jpxWAk"},
        "child": {
            "name": "p",
            # "class_": "BaseText-fEwdHD InstructionHed-cAxIoG InstructionHedSmall-jjUeNB fddHZs eFqQts",
        },
    },
    "desc": {
        "parent": {
            "name": "div",
            "class_": "GridItem-bufwYC dQxGga grid--item grid-layout__content",
        },
        "child": {"name": "div", "class_": "container--body-inner"},
    },
    "date": {
        "parent": {"name": "div", "class_": "content-header-text"},
        "child": {"name": "time", "attrs": {"data-testid": "ContentHeaderPublishDate"}},
    },
    "blank": {"parent": {}, "child": {}},
}


# fmt: off
INGREDIENTS = [
    'almond', 'amaretto', 'anchovy', 'anise', 'apple', 'apple juice',
    'apricot', 'artichoke', 'arugula', 'asian pear', 'asparagus', 'avocado',
    'bacon', 'banana', 'barley', 'basil', 'bean', 'beef', 'beef rib',
    'beef shank', 'beef tenderloin', 'beer', 'beet', 'bell pepper', 'berry',
    'biscuit', 'bitters', 'blackberry', 'blue cheese', 'blueberry',
    'bok choy', 'bourbon', 'bran', 'brandy', 'bread', 'breadcrumbs', 'brie',
    'brisket', 'broccoli', 'broccoli rabe', 'brown rice', 'brussel sprout',
    'bulgur', 'butter', 'buttermilk', 'butternut squash', 'cabbage', 'cake',
    'calvados', 'campari', 'candy', 'candy thermometer', 'cantaloupe',
    'capers', 'caraway', 'cardamom', 'carrot', 'cashew', 'caviar', 'celery',
    'chambord', 'champagne', 'chard', 'chartreuse', 'cheddar', 'cheese',
    'cherry', 'chestnut', 'chicken', 'chickpea', 'chile', 'chile pepper',
    'chili', 'chocolate', 'cinnamon', 'citrus', 'clam', 'clove', 'coconut',
    'cod', 'coffee', 'coffee grinder', 'coriander', 'corn', 'cornmeal',
    'cottage cheese', 'couscous', 'crab', 'cranberry', 'cranberry sauce',
    'cream cheese', 'cucumber', 'cumin', 'currant', 'curry', 'custard',
    'dairy', 'date', 'dill', 'dried fruit', 'duck', 'egg', 'egg nog',
    'eggplant', 'endive', 'escarole', 'fennel', 'feta', 'fig', 'fish',
    'flat bread', 'fontina', 'fortified wine', 'frangelico', 'fruit',
    'fruit juice', 'garlic', 'gin', 'ginger', 'goat cheese', 'goose',
    'gouda', 'grains', 'grape', 'grapefruit', 'grappa', 'green bean',
    'green onion/scallion', 'ground beef', 'ground lamb', 'guava', 'halibut',
    'ham', 'hazelnut', 'hominy/cornmeal/masa', 'honey', 'honeydew',
    'horseradish', 'hot pepper', 'hummus', 'ice cream', 'ice cream machine',
    'iced coffee', 'iced tea', 'jam or jelly', 'jerusalem artichoke',
    'kale', 'kirsch', 'kiwi', 'kumquat', 'lamb', 'lamb chop', 'lamb shank',
    'leek', 'legume', 'lemon', 'lemon juice', 'lemongrass', 'lentil',
    'lettuce', 'lima bean', 'lime', 'lime juice', 'lingonberry', 'liqueur',
    'lobster', 'lychee', 'macadamia nut', 'macaroni and cheese', 'mango',
    'maple syrup', 'margarita', 'marsala', 'marscarpone', 'martini',
    'mayonnaise', 'meat', 'melon', 'mezcal', 'milk/cream', 'mint',
    'molasses', 'monterey jack', 'mozzarella', 'muffin', 'mushroom',
    'mussel', 'mustard', 'mustard greens', 'nectarine', '"no meat', 'noodle',
    'nut', 'nutmeg', 'oat', 'oatmeal', 'octopus', 'okra', 'olive', 'onion',
    'orange', 'orange juice', 'oregano', 'orzo', 'oyster', 'pancake',
    'papaya', 'paprika', 'parmesan', 'parsley', 'parsnip', 'passion fruit',
    'pasta', 'pastry', 'pea', 'peach', 'peanut', 'peanut butter', 'pear',
    'pecan', 'pepper', 'pernod', 'persimmon', 'phyllo/puff pastry dough',
    'pickles', 'pine nut', 'pineapple', 'pistachio', 'plantain', 'plum',
    'poblano', 'pomegranate', 'pomegranate juice', 'poppy', 'pork',
    'pork chop', 'pork rib', 'pork tenderloin', 'port', 'potato',
    'potato salad', 'poultry', 'poultry sausage', 'prosciutto', 'prune',
    'pumpkin', 'punch', 'quail', 'quince', 'quinoa', 'rabbit', 'rack of lamb',
    'radicchio', 'radish', 'raisin', 'ramekin', 'raspberry', 'red wine',
    'rhubarb', 'rice', 'ricotta', 'rosemary', 'rum', 'rutabaga', 'rye',
    'saffron', 'sage', 'sake', 'salad', 'salad dressing', 'salmon', 'salsa',
    'sangria', 'sardine', 'sausage', 'scallop', 'scotch', 'semolina',
    'sesame', 'sesame oil', 'shallot', 'shellfish', 'sherry', 'shrimp',
    'snapper', 'soup/stew', 'sour cream', 'soy', 'soy sauce',
    'sparkling wine', 'spice', 'spinach', 'squash', 'squid', 'steak',
    'stew', 'stock', 'strawberry', 'sugar snap pea', 'sweet potato/yam',
    'swiss cheese', 'swordfish', 'taco', 'tamarind', 'tangerine', 'tapioca',
    'tart', 'tea', 'tequila', 'thyme', 'tilapia', 'tofu', 'tomatillo',
    'tomato', 'tortillas', 'tree nut', 'triple sec', 'tropical fruit',
    'trout', 'tuna', 'turnip', 'vanilla', 'veal', 'venison', 'vermouth',
    'vinegar', 'vodka', 'walnut', 'wasabi', 'watercress', 'watermelon',
    'whiskey', 'white wine', 'wild rice', 'wine', 'yogurt', 'yuca',
    'zucchini', 'turkey'
]
# fmt: on/
