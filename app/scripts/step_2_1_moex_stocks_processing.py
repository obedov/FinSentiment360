import pandas as pd
import chardet

# Список ценных бумаг, допущенных к торгам по состоянию на 08.04.2024
# https://www.moex.com/ru/listing/securities-list.aspx

# Загрузка исходного файла
file_path = '../data/raw/moex_stocks/exported_moex_stocks.csv'

# Определение кодировки файла с помощью chardet
with open(file_path, 'rb') as file:
    raw_data = file.read(100000)  # Читаем первые 100000 байт для определения кодировки
    encoding = chardet.detect(raw_data)['encoding']

# Чтение файла с автоматически определённой кодировкой
df = pd.read_csv(file_path, encoding=encoding)

# Посчитаем количество различных типов инструментов и выведем их
print(df['INSTRUMENT_TYPE'].value_counts())

# Выбор необходимых столбцов
columns_of_interest = ['TRADE_CODE', 'ISIN', 'EMITENT_FULL_NAME', 'ISSUE_AMOUNT', 'LIST_SECTION', 'INSTRUMENT_TYPE', 'INSTRUMENT_CATEGORY']
filtered_df = df[columns_of_interest]

# Фильтрация записей, где тикеры (TRADE_CODE) присутствуют
filtered_df = filtered_df.dropna(subset=['TRADE_CODE'])

# Дополнительная фильтрация по типам инструментов
instrument_types_of_interest = [
    "Акция привилегированная",
    "Акция обыкновенная",
    "Депозитарные расписки иностранного эмитента на акции",
    "Акции иностранного эмитента"
]
filtered_df = filtered_df[filtered_df['INSTRUMENT_TYPE'].isin(instrument_types_of_interest)]


# Сохранение отфильтрованных данных в новый файл с кодировкой UTF-8
filtered_file_path = '../data/processed/processed_moex_stocks.csv'

filtered_df.to_csv(filtered_file_path, index=False, encoding='utf-8')

print("Файл с отфильтрованными данными успешно сохранен в кодировке UTF-8:", filtered_file_path)
