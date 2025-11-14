import json
from elasticsearch import Elasticsearch
import pandas as pd

ES_HOST = "http://localhost:9200"
INDEX_NAME = "opennet_news"

es = Elasticsearch(ES_HOST)

def search(query, size=10):
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"]
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body, size=size)
    results = []
    for h in res.get("hits", {}).get("hits", []):
        results.append({
            "relevance": None, 
            "id": h["_id"],
            "title": h["_source"]["title"],
            "content": h["_source"]["content"]
        })
    return results

if __name__ == "__main__":
    queries = [
        "прошивки bios",
        "видеокарта nvidia",
        "программирование rust",
        "apache сервер",
        "linux wine",
        "chrome os установка",
        "процессоры intel",
        "redhat systemd",
        "генераторы сертификатов ssl",
        "мобильное приложение"
    ]

    all_results = {}
    excel_rows = []

    for q in queries:
        hits = search(q, size=10)
        all_results[q] = hits
        for hit in hits:
            excel_rows.append({
                "relevance": None,
                "query": q,
                "id": hit["id"],
                "title": hit["title"],
                "content": hit["content"]
            })

    with open("serp_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(excel_rows)
    df.to_excel("serp_results.xlsx", index=False)

    print("SERP сохранена в serp_results.json и serp_results.xlsx.")
