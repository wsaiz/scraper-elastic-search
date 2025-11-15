import requests
from elasticsearch import Elasticsearch

ES_HOST = "http://localhost:9200"
INDEX_NAME = "opennet_news"

es = Elasticsearch(ES_HOST)

if not es.ping():
    print("[ERROR] Не удалось подключиться к Elasticsearch")
    exit(1)

print(f"[INFO] Подключено к Elasticsearch: {ES_HOST}, индекс: {INDEX_NAME}")

def correct_spelling(text: str) -> str:
    url = "https://speller.yandex.net/services/spellservice.json/checkText"
    params = {"text": text, "lang": "ru,en"}
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        corrections = resp.json()
        if not corrections:
            return text
        for corr in reversed(corrections):
            word = text[corr['pos']:corr['pos']+corr['len']]
            suggestion = corr['s'][0] if corr.get('s') else word
            text = text[:corr['pos']] + suggestion + text[corr['pos']+corr['len']:]
        return text
    except Exception as e:
        print(f"[WARN] Не удалось исправить опечатки: {e}")
        return text


def search(query: str, size: int = 10):
    corrected_query = correct_spelling(query)
    if corrected_query != query:
        print(f"[INFO] Исправленный запрос: {corrected_query}")

    body = {
        "query": {
            "multi_match": {
                "query": corrected_query,
                "fields": ["title^2", "content"]  
            }
        }
    }

    res = es.search(index=INDEX_NAME, body=body, size=size)
    hits = res.get("hits", {}).get("hits", [])
    return hits


if __name__ == "__main__":
    print("Введите поисковый запрос (или 'exit' для выхода):")
    while True:
        query = input("> ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        size = 10
        if "," in query:
            try:
                query, size_str = query.split(",", 1)
                size = int(size_str)
            except:
                size = 10
            query = query.strip()

        results = search(query, size=size)
        if not results:
            print("[INFO] Результатов не найдено.")
            continue

        print(f"[INFO] Найдено {len(results)} результатов:\n")
        for i, hit in enumerate(results, 1):
            source = hit["_source"]
            title = source.get("title", "")
            content = source.get("content", "")
            url = source.get("url", "")
            keywords = source.get("keywords", [])
            keywords_str = ", ".join(keywords) if keywords else "-"

            print(
                f"[Result {i}]\n"
                f"   [Title]: {title}\n"
                f"   [URL]: {url}\n"
                f"   [Keywords]: {keywords_str}\n"
                f"   [Content]: {content[:500]}{'...' if len(content) > 500 else ''}\n"
            )
