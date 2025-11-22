import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib

xlsx_path = "serp_results_5000_with_relevance.xlsx"
df = pd.read_excel(xlsx_path)

required_columns = ['id', 'query', 'title', 'keywords', 'content', 'relevance']
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
df['query_title'] = df['query'] + " " + df['title']
df['query_keywords'] = df['query'] + " " + df['keywords']
df['query_content'] = df['query'] + " " + df['content_short']

# числовые признаки
df['title_len'] = df['title'].str.len()
df['keywords_count'] = df['keywords'].str.count(',') + 1  
df['content_len'] = df['content_short'].str.len()

numeric_features = ['title_len', 'keywords_count', 'content_len']
text_features = ['query_title', 'query_keywords', 'query_content']

df = df.drop_duplicates(subset='id')
df_sorted = df.sort_values('id')
test_fraction = 0.25
split_index = int(len(df_sorted) * (1 - test_fraction))

train_df = df_sorted.iloc[:split_index].copy()
test_df = df_sorted.iloc[split_index:].copy()

train_ids = set(train_df['id'])
test_ids = set(test_df['id'])
assert len(train_ids & test_ids) == 0, "id присутствуют и в train, и в test"

X_train = train_df[text_features + numeric_features]
y_train = train_df['relevance']
X_test = test_df[text_features + numeric_features]
y_test = test_df['relevance']

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
print("Модель сохранена в relevance_classifier.pkl")

