import json
import pandas as pd
import requests
from elasticsearch import Elasticsearch
from ranker import relevance_ranker

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
        for corr in reversed(corrections):
            word = text[corr['pos']:corr['pos']+corr['len']]
            suggestion = corr['s'][0] if corr.get('s') else word
            text = text[:corr['pos']] + suggestion + text[corr['pos']+corr['len']:]
        return text
    except:
        return text

def search(query: str, size: int = 10, ml_weight=0.7, es_weight=0.3):
    corrected_query = correct_spelling(query)

    body = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"title": {"query": corrected_query, "boost": 4}}},
                    {"match": {"keywords": {"query": corrected_query, "boost": 3}}},
                    {"match": {"content": {"query": corrected_query, "boost": 1}}},
                    {"match_phrase": {"content": {"query": corrected_query, "slop": 2, "boost": 5}}}
                ],
                "minimum_should_match": 1
            }
        }
    }

    res = es.search(index=INDEX_NAME, body=body, size=size)
    enhanced_res = ranker.rerank_results(corrected_query, res, ml_weight=ml_weight, es_weight=es_weight)
    hits = enhanced_res.get("hits", {}).get("hits", [])

    results = []
    for h in hits:
        source = h["_source"]
        results.append({
            "relevance": None,
            "id": h["_id"],
            "keywords": source.get("keywords", []),
            "title": source.get("title", ""),
            "content": source.get("content", "")  
        })
    return hits, results

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

    all_results_json = {}
    excel_rows = []

    for q in queries:
        print(f"[INFO] Поиск по запросу: {q}")
        hits, results_for_json = search(q, size=10)
        all_results_json[q] = results_for_json

        for h in hits:
            source = h["_source"]
            excel_rows.append({
                "relevance": None,
                "query": q,
                "id": h["_id"],
                "keywords": ", ".join(source.get("keywords", [])),
                "title": source.get("title", ""),
                "content": source.get("content", ""),  
                "ml_score": h.get("_ml_score", 0),
                "combined_score": h.get("_combined_score", 0)
            })

    with open("serp_results_ml.json", "w", encoding="utf-8") as f:
        json.dump(all_results_json, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(excel_rows)
    df.to_excel("serp_after_ml.xlsx", index=False)

    print("[INFO] SERP с ML сохранена в serp_results_ml.json и serp_after_ml.xlsx.")
