from mimetypes import init
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import re, time, sqlite3, requests
import logging
import random

from config import (
    PARSING_URL_LOG_FILE,
    RECIPES_DB_PATH,
    RECIPES_JSON_PATH,
    HEADERS,
    TAGS_INFO,
    PARSING_NUMBER_ATTEMPTS,
    SEARCH_URL,
    INIT_DB,
)

logging.basicConfig(
    filename="parse_url.log",
    filemode="a",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # %(name)s покажет, откуда лог
)
# logging.basicConfig(
#     level=logging.INFO,
#     encoding="utf-8",
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("parse_url.log", mode="a")],  # Только в файл!
# )

logger = logging.getLogger(__name__)


ERROR_REQUEST = "Запрос не успешен code возврата не равен 200"
ERROR_ATTEMPT_MAIN = "Выполнено большое количество запросов для поиска:"
ERROR_ATTEMPT_RECIPE = "Выполнено большое количество запросов для рецепта:"
ERROR_FORBIDDEN_403 = "Содержимое html содержит 403 Forbidden"


def check_text_in_tag(tags_dict, text, soup):
    result = False
    parent_tag = soup.find(**tags_dict["parent"])
    if parent_tag:
        child_tag = parent_tag.find(**tags_dict["child"])
        if parent_tag and child_tag:
            result = child_tag.text == text
    return result


def check_text_set(tags_dict, text_set, soup):
    result = False
    parent_tag = soup.find(**tags_dict["parent"])
    if parent_tag:
        search_list = parent_tag.find_all(**tags_dict["child"])
        if parent_tag and search_list:
            search_set = set([item.text for item in search_list])
            result = text_set == search_set
    return result


def check_desc(tags_dict, text, soup):
    result = False
    parent_tag = soup.find(**tags_dict["parent"])
    if parent_tag:
        child_tag = parent_tag.find(**tags_dict["child"])
        if child_tag:
            result = text in child_tag.text
    return result


def check_date(tags_dict, target_time, soup):
    result = False
    parent_tag = soup.find(**tags_dict["parent"])
    if parent_tag:
        time_tag = parent_tag.find(**tags_dict["child"])
        datetime_value = time_tag["datetime"]
        if datetime_value and target_time:
            page_timestamp = pd.to_datetime(datetime_value)
            result = target_time == page_timestamp
    return result


def check_row(row, soup):
    results = []
    results.append(check_date(TAGS_INFO["date"], row["date"], soup))
    results.append(check_text_in_tag(TAGS_INFO["title"], row["title"].strip(), soup))
    if not pd.isna(row["desc"]):
        results.append(check_desc(TAGS_INFO["desc"], row["desc"].strip(), soup))

    direction_set = set(row["directions"])
    results.append(check_text_set(TAGS_INFO["directions"], direction_set, soup))
    ingredients_set = set(row["ingredients"])
    results.append(check_text_set(TAGS_INFO["ingredients"], ingredients_set, soup))

    return all(results)


def check_403(soup):
    head = soup.find("head")
    if head.text == "403 Forbidden":
        raise Exception(ERROR_FORBIDDEN_403)


def prepare_json_df(df):
    df = df.dropna(how="all")
    df_cols = ["fat", "calories", "protein", "rating", "sodium", "title", "desc"]
    df["index_row"] = df.index
    df = df.drop_duplicates(subset=df_cols)
    return df


def get_content_from_url(url):
    resp = requests.get(url, headers=HEADERS, timeout=(20, 30))
    return resp.content


def get_url_content_iter(url):

    for attempt in range(1, PARSING_NUMBER_ATTEMPTS + 1):
        try:
            # Если 3 попытки не успешны выбрасываем исключение
            content = get_content_from_url(url)
            break
        except Exception as ex:
            logger.error(f"Попытка {attempt} не удалась. Ошибка: {str(ex)}")
            if attempt < PARSING_NUMBER_ATTEMPTS:
                time.sleep(2)
    else:
        desc_str = f" выполнено {PARSING_NUMBER_ATTEMPTS}"
        raise Exception(ERROR_ATTEMPT_MAIN + desc_str)

    return content


def get_full_epicurious_url(url):
    return "https://epicurious.com" + url


def get_link(links, row):

    result = ""
    for link in links:
        url_recipe = get_full_epicurious_url(link)
        # url_recipe = ""
        # для каждой проверяем, чтобы совпадал title и делаем проверку строки для этой ссылки
        # получаем ссылки с рецептов с поисковой страницы 3 попытки.
        content = get_url_content_iter(url_recipe)
        soup = BeautifulSoup(content, "html.parser")
        if check_row(row, soup):
            # если ссылка найдена присваиваем результат break
            result = link
            break
    return result


def init_database():
    """Удаляет старую таблицу и создает новую с нуля."""
    conn = sqlite3.connect(RECIPES_DB_PATH)
    cursor = conn.cursor()

    # Полностью удаляем старую таблицу, если она была
    cursor.execute("DROP TABLE IF EXISTS parsed_recipes")

    # Создаем новую таблицу заново
    cursor.execute("""
        CREATE TABLE parsed_recipes (
            df_index INTEGER PRIMARY KEY,
            title TEXT,
            recipe_url TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.debug("База данных успешно пересоздана (старые данные удалены).")


def main(init_db=False):
    logger.info("Start parsing")
    logger.debug("Read json file with pandas.")
    df = pd.read_json(RECIPES_JSON_PATH)
    df = prepare_json_df(df)
    if init_db:
        logger.info("Начинаем инициализацию БД.")
        init_database()
        logger.debug("Инициализация БД завершeна.")

    # df = df.iloc[37:40]

    conn = sqlite3.connect(RECIPES_DB_PATH)
    cursor = conn.cursor()
    try:
        logger.debug("Читаем уже записанные данные")
        cursor.execute("SELECT df_index FROM parsed_recipes")
        already_parsed_indices = [row[0] for row in cursor.fetchall()]

        if already_parsed_indices:
            logger.debug(
                f"Найдено прошлых результатов в БД: {len(already_parsed_indices)}. Пропускаем их."
            )
            # errors='ignore' нужен на случай, если в БД записаны индексы, которых нет в текущем срезе
            df = df.drop(index=already_parsed_indices, errors="ignore")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # tqdm.write("Начинаем обработку")
            logger.debug("Начинаем обработку")
            final_url = ""
            title = row["title"]
            url = SEARCH_URL(title=title)
            search_content = get_content_from_url(url)
            soup = BeautifulSoup(search_content, "html.parser")
            check_403(soup)
            logger.debug("Поисковый запрос выполнен.")
            recipe_titles = soup.find_all(
                "a", class_="BaseLink-haTOKm Link-ckAjCj fSXnEB"
            )
            recipe_links = [
                recipe["href"]
                for recipe in recipe_titles
                if recipe.h2.text == row["title"].strip()
            ]
            final_url = get_link(recipe_links, row)
            cursor.execute(
                """
                INSERT OR REPLACE INTO parsed_recipes (df_index, title, recipe_url)
                VALUES (?,  ?, ?)
            """,
                (idx, title, final_url),
            )
            conn.commit()
            if final_url:
                logger.debug(f"-> Успешно записано в БД. URL: {final_url}")
            else:
                logger.info(f"-> Ссылка для рецепта '{title}' не найдена.")
            delay = random.uniform(0.1, 2.0)
            # delay = 1.5
            time.sleep(delay)
            # добавляем результат
    except Exception as ex:
        conn.close()
        if ERROR_ATTEMPT_MAIN in str(ex):
            logger.error(f"Парсинг остановлен. {str(ex)}, {idx}, {row}")
        elif ERROR_ATTEMPT_RECIPE in str(ex):
            logger.error(f"Парсинг остановлен. {str(ex)}, {idx}, {row}")
        elif ERROR_FORBIDDEN_403 in str(ex):
            logger.error(f"Парсинг остановлен. {str(ex)}, {idx}, {row}")
        else:
            logger.error(f"Парсинг остановлен. {str(ex)}, {idx}, {row}")


def process_manage(init_db=False):
    n = 200
    for i in range(n):
        main(init_db=init_db)
        time.sleep(10)


if __name__ == "__main__":
    process_manage(INIT_DB)
