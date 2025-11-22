import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib

xlsx_path = "serp_results_5000_with_relevance.xlsx"
df = pd.read_excel(xlsx_path)

required_columns = ['query', 'title', 'keywords', 'content', 'relevance']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"В файле нет колонки '{col}'")

print(f"Всего статей: {len(df)}")
print(f"Релевантных: {df['relevance'].sum()}")
print(f"Нерелевантных: {len(df)-df['relevance'].sum()}")


def get_first_two_sentences(text):
    sentences = str(text).split(".")
    first_two = ". ".join(sentences[:2]).strip()
    if not first_two.endswith("."):
        first_two += "."
    return first_two

df['content_short'] = df['content'].apply(get_first_two_sentences)

# комбинированные признаки
df['query_title'] = df['query'] + " " + df['title']
df['query_keywords'] = df['query'] + " " + df['keywords']
df['query_content'] = df['query'] + " " + df['content_short']

# числовые признаки
df['title_len'] = df['title'].str.len()
df['keywords_count'] = df['keywords'].str.count(',') + 1  
df['content_len'] = df['content_short'].str.len()

numeric_features = ['title_len', 'keywords_count', 'content_len']


text_features = ['query_title', 'query_keywords', 'query_content']
X = df[text_features + numeric_features]
y = df['relevance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        ('query_title_tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)), 'query_title'),
        ('query_keywords_tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,1)), 'query_keywords'),
        ('query_content_tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2)), 'query_content'),
        ('numeric', StandardScaler(), numeric_features)
    ]
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=10000, class_weight='balanced'))
])

print("Обучаем модель...")
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

report_dict = classification_report(y_test, y_pred, output_dict=True)

report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['accuracy', 'precision'] = accuracy
report_df.loc['accuracy', 'recall'] = accuracy
report_df.loc['accuracy', 'f1-score'] = accuracy
report_df.loc['accuracy', 'support'] = y_test.shape[0]

report_df.to_excel("model_training_stats.xlsx", index=True)
print("Статистика обучения модели сохранена в 'model_training_stats.xlsx'")


joblib.dump(model, "relevance_classifier.pkl")



def predict_relevance(new_articles):
    df_new = pd.DataFrame(new_articles)
    df_new['content_short'] = df_new['content'].apply(get_first_two_sentences)
    df_new['query_title'] = df_new['query'] + " " + df_new['title']
    df_new['query_keywords'] = df_new['query'] + " " + df_new['keywords']
    df_new['query_content'] = df_new['query'] + " " + df_new['content_short']
    df_new['title_len'] = df_new['title'].str.len()
    df_new['keywords_count'] = df_new['keywords'].str.count(',') + 1
    df_new['content_len'] = df_new['content_short'].str.len()

    X_new = df_new[text_features + numeric_features]
    probs = model.predict_proba(X_new)[:, 1]
    df_new['relevance_prob'] = probs
    df_new['relevance_pred'] = (probs >= 0.5).astype(int)
    return df_new

new_articles = [
    {"query": "linux firewall", "title": "Настройка iptables", "keywords": "firewall, linux", "content": "Полное руководство по iptables. С примерами настройки."},
    {"query": "linux firewall", "title": "Настройка ufw", "keywords": "firewall, ufw", "content": "UFW — простой способ настройки firewall. Примеры использования."}
]

df_pred = predict_relevance(new_articles)
print(df_pred[['title', 'relevance_prob', 'relevance_pred']])
