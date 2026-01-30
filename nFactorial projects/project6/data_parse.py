import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

session = requests.Session()
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def get_article_details(url):
    try:
        res = session.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')

        date_tag = soup.find('meta', itemprop='dateModified')
        date_iso = date_tag['content'] if date_tag else None

        text_block = soup.find('div', class_='content_main_text', itemprop='articleBody')
        content = ""
        if text_block:
            paragraphs = text_block.find_all('p')
            content = " ".join([p.text.strip() for p in paragraphs])

        tags = []
        tags_container = soup.find('div', class_='content_main_text_tags', id='comm')
        if tags_container:
            tag_links = tags_container.find_all('a')
            tags = [tag.text.strip() for tag in tag_links]

        return content, ", ".join(tags), date_iso

    except Exception as e:
        return "", "", None

def process_article(item):
    try:
        title_tag = item.find('span', class_='content_main_item_title')
        if not title_tag: return None

        title = title_tag.text.strip()
        link_tag = item.find('a')
        if not link_tag: return None

        link = "https://tengrinews.kz" + link_tag['href']

        content, tags, date_val = get_article_details(link)

        return {
            'title': title,
            'link': link,
            'content': content,
            'tags': tags,
            'date': date_val
        }
    except:
        return None

def parse_tengri_parallel(page_start = 1, pages_limit=1000):
    full_data = []

    for page in range(page_start, pages_limit + 1):
        print(f"Парсим страницу {page}...")
        url = f"https://tengrinews.kz/news/page/{page}/"

        try:
            response = session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('div', class_='content_main_item')

            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(process_article, articles))

            full_data.extend([r for r in results if r is not None])

        except Exception as e:
            print(f"Ошибка на странице {page}: {e}")

    return full_data


start_time = time.time()

data = parse_tengri_parallel(pages_limit=5000)

df = pd.DataFrame(data)
df['tags'] = df['tags'].apply(lambda x: x.split(', ') if x else [])
df.to_parquet('tengri_data.parquet')
df.to_json('tengri_data.jsonl', orient='records', lines=True, force_ascii=False)
df.to_csv('tengri_full.csv', index=False, encoding='utf-8-sig')

end_time = time.time()
duration = (end_time - start_time) / 60
print(f"Сбор завершен за {duration:.2f} минут!")