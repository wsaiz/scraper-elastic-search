import pandas as pd

EXCEL_FILE = "serp_after_ml.xlsx"

df = pd.read_excel(EXCEL_FILE)
df['relevance'] = df['relevance'].fillna(0).astype(int)


def precision_at_k(relevance_list, k):
    relevance_list = relevance_list[:k]
    return sum(relevance_list) / k


def average_precision(relevance_list):
    relevant_docs = 0
    score = 0.0
    for i, rel in enumerate(relevance_list, start=1):
        if rel == 1:
            relevant_docs += 1
            score += relevant_docs / i
    if relevant_docs == 0:
        return 0.0
    return score / relevant_docs


def reciprocal_rank(relevance_list):
    for i, rel in enumerate(relevance_list, start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0


queries = df['query'].unique()
results = []

for q in queries:
    rels = df[df['query'] == q]['relevance'].tolist()

    results.append({
        'query': q,
        'Precision@5': precision_at_k(rels, 5),
        'Precision@10': precision_at_k(rels, 10),
        'AveragePrecision': average_precision(rels),
        'MRR': reciprocal_rank(rels)
    })

results_df = pd.DataFrame(results)

summary_df = pd.DataFrame({
    'Metric': ['MAP', 'Mean Precision@5', 'Mean Precision@10', 'Mean MRR'],
    'Value': [
        results_df['AveragePrecision'].mean(),
        results_df['Precision@5'].mean(),
        results_df['Precision@10'].mean(),
        results_df['MRR'].mean()
    ]
})

with pd.ExcelWriter("serp_metrics_after_ml.xlsx") as writer:
    results_df.to_excel(writer, sheet_name="metrics", index=False, startrow=0)
    summary_df.to_excel(writer, sheet_name="metrics", index=False, startrow=len(results_df) + 3)

print("Метрики сохранены в serp_metrics_after_ml.xlsx.")
