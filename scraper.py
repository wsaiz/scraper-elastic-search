import requests
from bs4 import BeautifulSoup
import time
import json
import re
import html
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Optional, List, Dict


BASE_URL = "https://www.opennet.ru/opennews/art.shtml?num={}"

INVISIBLE = {
    "\u200B", "\u200C", "\u200D", "\uFEFF", "\u00AD"
}

def remove_control_chars(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if code < 32 or code == 127:
            continue
        if ch in INVISIBLE:
            continue
        out.append(ch)
    return "".join(out)

def normalize_spaces(text: str) -> str:
    result = []
    was_space = False
    for ch in text:
        if ch.isspace():
            was_space = True
        else:
            if was_space and result:
                result.append(" ")
            result.append(ch)
            was_space = False
    return "".join(result).strip()

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = fix_word_glues(text)
    text = remove_control_chars(text)
    text = normalize_spaces(text)

    return text

def is_lat(ch):
    return 'A' <= ch <= 'Z' or 'a' <= ch <= 'z'

def is_cyr(ch):
    return ('А' <= ch <= 'Я') or ('а' <= ch <= 'я') or ch in "ёЁ"

def is_digit(ch):
    return '0' <= ch <= '9'

def fix_word_glues(text: str) -> str:
    out = []
    i = 0
    L = len(text)

    while i < L:
        ch = text[i]
        if i+1 < L and ch == '.' and (i > 0) and text[i-1].isalnum() and text[i+1].isupper():
            out.append('. ')
            i += 1
            continue

        if ch == ')' and i+1 < L and text[i+1].isupper():
            out.append(') ')
            i += 1
            continue

        if is_cyr(ch) and i+1 < L and is_lat(text[i+1]):
            j = i+2
            while j < L and is_lat(text[j]):
                j += 1
            out.append(ch + " ")
            i += 1
            continue

        
        if is_lat(ch) and i+1 < L and is_cyr(text[i+1]):

            j = i+1
            while j < L and is_lat(text[j]):
                j += 1
            out.append(ch + " ")
            i += 1
            continue

        if is_digit(ch) and i > 0 and (is_lat(text[i-1]) or is_cyr(text[i-1])):
            prev = text[i-1]
            if not (prev in "Dd"):
                out.append(" " + ch)
                i += 1
                continue

        if i+1 < L and is_digit(text[i+1]) and (is_lat(ch) or is_cyr(ch)):
            if ch not in "Dd":
                out.append(ch + " ")
                i += 1
                continue

        out.append(ch)
        i += 1

    return normalize_spaces("".join(out))

def insert_spaces_around_tags(html: str) -> str:
    out = []
    L = len(html)
    i = 0

    while i < L:
        ch = html[i]


        if ch == '<' and i > 0 and html[i-1].isalnum():
            out.append(" <")
            i += 1
            continue

        if ch == '>' and i+1 < L and html[i+1].isalnum():
            out.append("> ")
            i += 1
            continue

        out.append(ch)
        i += 1

    text = "".join(out)

    text = fix_word_glues(text)

    return text



def extract_article_text(soup: BeautifulSoup) -> Optional[str]:
    td = soup.select_one("table.ttxt2 td.chtext")
    text = None
    if td:
        raw_html = str(td)
        raw_html = insert_spaces_around_tags(raw_html)
        text = BeautifulSoup(raw_html, "lxml").get_text(" ", strip=True)
        text = clean_text(text)

    if not text or len(text) < 30:
        meta = soup.select_one("meta[property='og:description']")
        if meta and meta.get("content"):
            text = clean_text(meta["content"])

    return text if text and len(text) >= 30 else None


def extract_article_title(soup: BeautifulSoup, num: int) -> str:
    span = soup.select_one("span#r_title")
    if span:
        return clean_text(span.get_text(" ", strip=True))

    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))

    h2 = soup.find("h2")
    if h2:
        return clean_text(h2.get_text(" ", strip=True))

    return f"Новость {num}"

def extract_article_keywords(soup: BeautifulSoup) -> List[str]:
    keywords = []
    span = soup.select_one("span#r_keyword_link")
    if span:
        links = span.find_all("a")
        for a in links:
            kw = clean_text(a.get_text(strip=True))
            if kw:
                keywords.append(kw)
    return keywords




def fetch_article(num: int, delay: float = 0.1) -> Optional[Dict]:
    url = BASE_URL.format(num)
    print(f"[REQ] {num}: {url}")

    if delay > 0:
        time.sleep(delay)

    try:
        r = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {num}: {e}")
        return None

    if r.status_code != 200:
        print(f"[ERROR] {num}: HTTP {r.status_code}")
        return None

    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, "lxml")

    text = extract_article_text(soup)
    if not text:
        return None

    title = extract_article_title(soup, num)
    keywords = extract_article_keywords(soup)

    return {
        "id": str(num),
        "url": url,
        "title": title,
        "content": text,
        "keywords": keywords,  
    }



def scrape_range(start_num: int, amount: int,
                 max_workers: int = 20, request_delay: float = 0.1) -> List[Dict]:
    nums = list(range(start_num, start_num - amount, -1))
    results = []
    results_lock = Lock()

    print(f"[SCRAPER] Потоки: {max_workers}, Статей: {amount}")

    def worker(num):
        article = fetch_article(num, delay=request_delay)
        if article:
            with results_lock:
                results.append(article)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, num): num for num in nums}

        for i, future in enumerate(as_completed(futures), start=1):
            if i % 10 == 0:
                print(f"[PROGRESS] {i}/{len(nums)}")


    results.sort(key=lambda x: int(x['id']), reverse=True)
    print(f"[DONE] Собрано статей: {len(results)}")
    return results



def save_json(data, filename="opennet_news.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[FILE] Сохранено: {filename}")



if __name__ == "__main__":
    START_NUM = 64251
    LIMIT = 10000
    MAX_WORKERS = 20
    REQUEST_DELAY = 0.1

    t0 = time.time()

    news = scrape_range(
        START_NUM,
        LIMIT,
        max_workers=MAX_WORKERS,
        request_delay=REQUEST_DELAY
    )

    save_json(news)
    print(f"[TIME] {time.time() - t0:.2f} сек.")
