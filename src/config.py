from pathlib import Path
import logging

PARSING_NUMBER_ATTEMPTS = 3
RECIPES_JSON_PATH = Path("src/data/full_format_recipes.json")
RECIPES_DB_PATH = Path("src/data/recipe_urls.db")
PARSING_URL_LOG_FILE = Path("src/data/recipe_urls.db")

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
