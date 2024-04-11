import pickle
from sentence_transformers import SentenceTransformer
from app.data.criterions.criterions_news import criterions_news
from app.data.criterions.criterions_analysis import criterions_analysis


model_names = [
    'distiluse-base-multilingual-cased-v2',
    'paraphrase-multilingual-MiniLM-L12-v2'
]

for model_name in model_names:
    model = SentenceTransformer(model_name)
    embeddings_news = model.encode(criterions_news, convert_to_tensor=True)
    embeddings_analysis = model.encode(criterions_analysis, convert_to_tensor=True)

    # Сохранение эмбеддингов для новостей
    with open(f'../data/embeddings/embeddings_news_{model_name.replace("/", "_")}.pkl', 'wb') as f:
        pickle.dump(embeddings_news, f)

    # Сохранение эмбеддингов для анализа
    with open(f'../data/embeddings/embeddings_analysis_{model_name.replace("/", "_")}.pkl', 'wb') as f:
        pickle.dump(embeddings_analysis, f)
