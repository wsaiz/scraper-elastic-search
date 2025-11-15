import os
import json
import logging
from elasticsearch import Elasticsearch, helpers


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ES_HOST = "http://localhost:9200"
INDEX_NAME = "opennet_news"
JSON_FILE = os.path.join(os.path.dirname(__file__), "..", "opennet_news.json")

es = Elasticsearch(ES_HOST)
if not es.ping():
    logger.error("Не удалось подключиться к Elasticsearch")
    exit(1)

def create_index(es_client, index_name):
    settings = {
        "settings": {
            "analysis": {
                "filter": {
                    "russian_stop": {"type": "stop", "stopwords": "_russian_"},
                    "russian_stemmer": {"type": "stemmer", "language": "russian"},
                    "russian_synonyms": {
                        "type": "synonym",
                        "synonyms": [
                            "linux, ubuntu, fedora, debian, centos, rhel, opensuse, suse, линукс, убунту, федора, дебян, центос, рел, опенсусе, сусе",
                            "windows, win, виндовс, вин, виндоус",
                            "macos, mac, apple, макос, мак, эппл",
                            "rust, раст, ржавый",
                            "python, py, пайтон, питон",
                            "javascript, js, джаваскрипт, жс, яваскрипт",
                            "java, джава, ява",
                            "gnome, гном",
                            "plasma, плазма",
                            "xfce, иксфце, иксфсе",
                            "desktop, рабочий стол, десктоп",
                            "wayland, вейленд, вяленый",
                            "firefox, фаерфокс",
                            "mozilla, мозилла",
                            "chrome, google chrome, хром, гугл хром",
                            "http, хттп",
                            "https, хттпс",
                            "apache, апаче",
                            "server, сервер",
                            "nvidia, нвидиа",
                            "intel, интел",
                            "gpu, графический процессор, видеокарта",
                            "cpu, центральный процессор, цпу, процессор, цп",
                            "vulkan, вулкан",
                            "opengl, опенгл",
                            "wine, вайн",
                            "systemd, системд",
                            "system, система",
                            "shell, шелл, командная строка, консоль",
                            "root, рут, администратор",
                            "postgresql, postgres, постгрескьюэл, постгрес, постгря",
                            "flatpak, флатпак",
                            "raspberry, raspberry pi, распберри, распберри пи",
                            "gplv, gpl, license, гпл, лицензия",
                            "live, iso, лайв, айсо"
                        ]
                    }
                },
                "analyzer": {
                    "russian_analyzer": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "russian_stop", "russian_stemmer", "russian_synonyms"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "url": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "russian_analyzer"},
                "content": {"type": "text", "analyzer": "russian_analyzer"},
                "keywords": {"type": "keyword"} 
            }
        }
    }

    if es_client.indices.exists(index=index_name):
        logger.info(f"[INFO] Индекс {index_name} уже существует, удаляем...")
        es_client.indices.delete(index=index_name)

    es_client.indices.create(index=index_name, body=settings)
    logger.info(f"[INFO] Индекс {index_name} создан успешно")


def index_documents(es_client, index_name, json_file):
    if not os.path.exists(json_file):
        logger.error(f"Файл не найден: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        news = json.load(f)

    actions = [
        {
            "_index": index_name,
            "_id": item["id"],
            "_source": {
                "id": item["id"],
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "keywords": item.get("keywords", []) 
            }
        }
        for item in news
    ]

    helpers.bulk(es_client, actions)
    logger.info(f"[INFO] Индексировано {len(actions)} документов")
    es_client.indices.refresh(index=index_name)



def main():
    create_index(es, INDEX_NAME)
    index_documents(es, INDEX_NAME, JSON_FILE)

if __name__ == "__main__":
    main()
