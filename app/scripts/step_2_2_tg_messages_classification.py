import pandas as pd
import re
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from typing import Tuple

# Загрузка данных
messages_df = pd.read_csv('../data/raw/tg_messages/tg_stock_messages.csv')
stocks_df = pd.read_csv('../data/processed/processed_moex_stocks.csv')
tickers_and_names = pd.concat([stocks_df['TRADE_CODE'], stocks_df['EMITENT_FULL_NAME']]).unique()

# Ключевые слова для определения рекламы
keywords_adv = ['реклама', 'скидка', 'акция', 'подписка']

# Список моделей для испытаний
model_names = [
    'paraphrase-multilingual-MiniLM-L12-v2',
    # 'distiluse-base-multilingual-cased-v2',
]


# Функция для поиска тикеров по хэштегу и в тексте
def get_tickers_info(message: str) -> Tuple[str, bool]:
    found_tickers = [ticker for ticker in tickers_and_names if ticker.lower() in message.lower()]
    hashtags = re.findall(r'#(\w+)', message)
    found_tickers.extend([hashtag.upper() for hashtag in hashtags if hashtag.upper() in tickers_and_names])
    return ', '.join(set(found_tickers)), len(found_tickers) > 1


# Загрузка и использование эмбеддингов для классификации
# Функция классификации сообщений и определения тикеров с батчами
def classify_batch(messages, model, criterion_embeddings_news, criterion_embeddings_analysis):
    categories = []
    tickers_info = []

    for message in messages:
        if any(keyword in message.lower() for keyword in keywords_adv):
            category = 'Реклама'
        else:
            message_embedding = model.encode(message, convert_to_tensor=True)
            similarity_news = util.pytorch_cos_sim(message_embedding, criterion_embeddings_news).mean()
            similarity_analysis = util.pytorch_cos_sim(message_embedding, criterion_embeddings_analysis).mean()

            category = 'Новость' if similarity_news > similarity_analysis else 'Мнение/Анализ'

        tickers_str, is_multi_ticker = get_tickers_info(message)
        categories.append(category)
        tickers_info.append((tickers_str, is_multi_ticker))

    return categories, tickers_info


# Обработка сообщений с использованием каждой модели и батчей
batch_size = 512
for model_name in model_names:
    model = SentenceTransformer(model_name)

    with open(f'../data/embeddings/embeddings_news_{model_name}.pkl', 'rb') as f:
        criterion_embeddings_news = pickle.load(f)
    with open(f'../data/embeddings/embeddings_analysis_{model_name}.pkl', 'rb') as f:
        criterion_embeddings_analysis = pickle.load(f)

    classified_results = []
    for start_index in tqdm(range(0, len(messages_df), batch_size), desc=f'Processing with {model_name}'):
        batch_messages = messages_df['Сообщение'][start_index:start_index + batch_size].tolist()
        batch_categories, batch_tickers_info = classify_batch(batch_messages, model, criterion_embeddings_news,
                                                              criterion_embeddings_analysis)

        for category, (tickers_str, is_multi_ticker) in zip(batch_categories, batch_tickers_info):
            classified_results.append((category, tickers_str, is_multi_ticker))

    classified_df = pd.DataFrame(classified_results, columns=['Категория', 'Тикеры', 'Мульти тикер'])
    messages_df[['Категория', 'Тикеры', 'Мульти тикер']] = classified_df

    # Сохранение результатов для текущей модели
    output_all_path = f'../data/processed/all_categorized_tg_messages_{model_name}.csv'
    messages_df.to_csv(output_all_path, index=False, encoding='utf-8')
    print(f"Все категоризированные данные для модели {model_name} сохранены в файле: {output_all_path}")

    # TODO если сообщение с категорией Новость, без тикеров (и без упоминаний отраслей), то не включать ее

    # Фильтрация и сохранение очищенных данных
    cleaned_messages_df = messages_df[(messages_df['Тикеры'] != '') | (messages_df['Категория'] != 'Не определено')]
    output_cleaned_path = f'../data/processed/cleaned_categorized_tg_messages_{model_name}.csv'
    cleaned_messages_df.to_csv(output_cleaned_path, index=False, encoding='utf-8')
    print(f"Очищенные и категоризированные данные для модели {model_name} сохранены в файле: {output_cleaned_path}")
