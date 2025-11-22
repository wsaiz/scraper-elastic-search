import pandas as pd
import os
from dotenv import load_dotenv
import time
from mistralai import Mistral

load_dotenv()
api_key = os.getenv("API_KEY")  
ai_model = "mistral-small-latest" 

xlsx_path_from = "serp_results_5000.xlsx"
xlsx_path_to = "serp_results_5000_with_relevance.xlsx"

def get_mistral_client():
    try:
        client = Mistral(api_key=api_key)
        return client
    except Exception as e:
        print(f"Ошибка при инициализации клиента Mistral: {e}")
        return None


def make_request(prompt):
    client = get_mistral_client()
    if not client:
        return None
    
    try:
        response = client.chat.complete(
            model=ai_model,
            messages=[
                {
                    "role": "system",
                    "content": """Ты оцениваешь релевантность статей по запросу.
Ответь только цифрами 1 (релевантно) или 0 (не релевантно) через запятую, строго в том порядке, как статьи в списке."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False
        )
        
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            return content
        else:
            print("Пустой ответ от API")
            return None
            
    except Exception as e:
        print(f"Ошибка при запросе к Mistral API: {e}")
        return None


def main():
    if not api_key:
        print("Ошибка: MISTRAL_API_KEY не найден в переменных окружения")
        return
    
    if not os.path.exists(xlsx_path_from):
        print(f"Файл {xlsx_path_from} не найден")
        return
    
    df = pd.read_excel(xlsx_path_from)
    print(f"Всего строк в файле: {len(df)}")

    query_ids = df['query'].unique()
    total_processed = 0

    print(f"Найдено уникальных запросов: {len(query_ids)}")

    for query in query_ids:
        print(f"\n{'='*50}")
        print(f" Обрабатываем запрос: {query}")
        articles = df[df['query'] == query]
        num_articles = len(articles)
        if articles['relevance'].notna().sum() == num_articles:
            print(f"Все {num_articles} статей уже обработаны, пропускаем")
            total_processed += num_articles
            continue

        prompt = f"Запрос: {query}\n\n"
        for i, (_, row) in enumerate(articles.iterrows(), 1):
            title = row['title']
            keywords = row['keywords'][:250]  
            content = row['content']
            first_two_sentences = ". ".join(content.split(".")[:2]).strip()
            if not first_two_sentences.endswith("."):
                first_two_sentences += "."

            prompt += f"{i}. title={title}. keywords={keywords}. content={first_two_sentences}\n"

        prompt += f"\nОтвет (только {num_articles} цифр через запятую):"

        try:
            response = make_request(prompt)
            
            if not response:
                print(f"Пустой ответ от API для запроса '{query}', пропускаем")
                continue
                
            print(f"Ответ модели: {response}")

            ratings = []
            for char in response:
                if char in '01':
                    ratings.append(int(char))
                if len(ratings) == num_articles:
                    break

            print(f"Спарсенные оценки: {ratings}")

            if len(ratings) == num_articles:
                for idx, rating in zip(articles.index, ratings):
                    df.at[idx, 'relevance'] = rating

                total_processed += num_articles
                print(f" Обработано статей: {num_articles} | Всего: {total_processed}")

                df.to_excel(xlsx_path_to, index=False)
                print("Промежуточный результат сохранен")
            else:
                print(f"Получено {len(ratings)} оценок, ожидалось {num_articles}")

            time.sleep(1)  

        except Exception as e:
            print(f"Ошибка при обработке запроса '{query}': {e}")

    print(f"\nОбработка завершена!")
    print(f"Результаты сохранены в: {xlsx_path_to}")

if __name__ == "__main__":
    main()