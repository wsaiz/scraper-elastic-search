import requests
from elasticsearch import Elasticsearch
from ranker import relevance_ranker 
import pprint

ES_HOST = "http://localhost:9200"
INDEX_NAME = "opennet_news"

es = Elasticsearch(ES_HOST)
if not es.ping():
    print("[ERROR] Не удалось подключиться к Elasticsearch")
    exit(1)
print(f"[INFO] Подключено к Elasticsearch: {ES_HOST}, индекс: {INDEX_NAME}")


ranker = relevance_ranker(model_path='./llm/relevance_classifier.pkl')

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

def search(query: str, size: int = 10, ml_weight=0.7, es_weight=0.3):
    """Поиск с ранжированием ML"""
    corrected_query = correct_spelling(query)
    if corrected_query != query:
        print(f"[INFO] Исправленный запрос: {corrected_query}")

    body = {
    "query": {
        "bool": {
            "should": [
                {
                    "match": {
                        "title": {
                            "query": corrected_query,
                            "boost": 4
                        }
                    }
                },
                {
                    "match": {
                        "keywords": {
                            "query": corrected_query,
                            "boost": 3
                        }
                    }
                },
                {
                    "match": {
                        "content": {
                            "query": corrected_query,
                            "boost": 1
                        }
                    }
                },
                {
                    "match_phrase": {
                        "content": {
                            "query": corrected_query,
                            "slop": 2,
                            "boost": 5
                        }
                    }
                }
            ],
            "minimum_should_match": 1
        }
    }
}


    res = es.search(index=INDEX_NAME, body=body, size=size)
    enhanced_res = ranker.rerank_results(corrected_query, res, ml_weight=ml_weight, es_weight=es_weight)
    hits = enhanced_res.get("hits", {}).get("hits", [])
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
            ml_score = hit.get("_ml_score", 0)
            combined_score = hit.get("_combined_score", 0)

            print(
                f"[Result {i}]\n"
                f"   [Title]: {title}\n"
                f"   [URL]: {url}\n"
                f"   [Keywords]: {keywords_str}\n"
                f"   [Content]: {content[:500]}{'...' if len(content) > 500 else ''}\n"
                f"   [ML Score]: {ml_score:.3f}, [Combined Score]: {combined_score:.3f}\n"
            )
