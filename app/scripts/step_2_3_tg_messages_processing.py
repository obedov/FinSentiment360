import pandas as pd
import spacy

# Загрузка русскоязычной NLP-модели
nlp = spacy.load("ru_core_news_sm")


# TODO поправить обработку для классифицированных сообщений

# Функция для вывода промежуточных результатов
def log_intermediate_results(df, step_description):
    print(f"{step_description}:")
    print(f"Количество строк: {len(df)}")
    print("Первые 5 строк:")
    print(df.head())
    print("-" * 50)  # Разделитель


# Загрузка данных
data_path = '../data/raw/tg_messages/tg_stock_messages.csv'
messages_df = pd.read_csv(data_path)

# Логирование исходных данных
log_intermediate_results(messages_df, "Исходные данные")


# Применение функции нормализации с использованием spaCy nlp.pipe для ускорения обработки
def normalize_texts(texts):
    processed_texts = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):  # Отключаем ненужные компоненты
        lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
        processed_text = ' '.join(lemmas)
        processed_texts.append(processed_text)
    return processed_texts


# Применение функции нормализации к каждому сообщению
messages_df['normalized_message'] = normalize_texts(messages_df['Сообщение'])

# Логирование результатов после нормализации
log_intermediate_results(messages_df, "После нормализации")

# Сохранение обработанных данных
output_path = '../data/processed/processed_tg_messages.csv'
messages_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Обработанные данные сохранены в файле: {output_path}")
