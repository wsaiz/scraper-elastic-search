import joblib
import pandas as pd

class relevance_ranker:
    def __init__(self, model_path='./ml/relevance_classifier.pkl'):

        self.model = joblib.load(model_path)

    def prepare_article_data(self, query_text, article_data):
        title = article_data.get('title', '')
        content = article_data.get('content', '')
        keywords = ', '.join(article_data.get('keywords', [])) if isinstance(article_data.get('keywords'), list) else article_data.get('keywords', '')

        df = pd.DataFrame([{
            'query': query_text,
            'title': title,
            'keywords': keywords,
            'content': content
        }])

        df['content_short'] = df['content'].apply(lambda x: '. '.join(str(x).split('.')[:2]).strip() + '.')
        df['query_title'] = df['query'] + ' ' + df['title']
        df['query_keywords'] = df['query'] + ' ' + df['keywords']
        df['query_content'] = df['query'] + ' ' + df['content_short']
        df['title_len'] = df['title'].str.len()
        df['keywords_count'] = df['keywords'].str.count(',') + 1
        df['content_len'] = df['content_short'].str.len()

        return df

    def calculate_ml_score(self, query_text, article_data):
        try:
            df = self.prepare_article_data(query_text, article_data)
            text_features = ['query_title', 'query_keywords', 'query_content']
            numeric_features = ['title_len', 'keywords_count', 'content_len']
            X = df[text_features + numeric_features]
            ml_score = self.model.predict_proba(X)[0, 1]
            return float(ml_score)
        except Exception as e:
            print(f"[ERROR] ML-ранжирование: {e}")
            return 0.0

    def rerank_results(self, query_text, es_results, ml_weight=0.7, es_weight=0.3):
        if not es_results or 'hits' not in es_results:
            return es_results

        hits = es_results['hits']['hits']
        es_scores = [hit['_score'] for hit in hits]
        max_es_score = max(es_scores) if es_scores else 1
        min_es_score = min(es_scores) if es_scores else 0
        
        enhanced_results = []

        for i, hit in enumerate(hits):
            article_data = {
                'title': hit['_source'].get('title', ''),
                'content': hit['_source'].get('content', ''),
                'keywords': hit['_source'].get('keywords', []),
                '_score': hit['_score']
            }

            ml_score = self.calculate_ml_score(query_text, article_data)
            
            if max_es_score > min_es_score:
                es_score_normalized = (hit['_score'] - min_es_score) / (max_es_score - min_es_score)
            else:
                es_score_normalized = 1.0
                
            combined_score = ml_weight * ml_score + es_weight * es_score_normalized

            enhanced_results.append({
                **hit,
                '_ml_score': ml_score,
                '_es_score_normalized': es_score_normalized,
                '_combined_score': combined_score
            })

        enhanced_results.sort(key=lambda x: x['_combined_score'], reverse=True)
        es_results['hits']['hits'] = enhanced_results
        return es_results
