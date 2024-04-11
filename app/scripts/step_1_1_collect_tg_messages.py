import asyncio
from telethon import TelegramClient, functions
import pandas as pd
from datetime import datetime, timezone
from decouple import config

system_version = '4.20.05-mars-tg'


def get_tg_api_id():
    return config('TELEGRAM_API_ID')


def get_tg_hash():
    return config('TELEGRAM_HASH')


api_id = get_tg_api_id()
api_hash = get_tg_hash()

tg_channels = [
    "https://t.me/profinvestr",
    "https://t.me/Information_disclosure",
    "https://t.me/regentus"
]


async def collect_all_tg_messages(channels):
    messages = []
    async with TelegramClient('mars', api_id, api_hash, system_version=system_version) as client:
        print("Подключение к Telegram...")
        await client.start()
        print("Подключено к Telegram.")

        start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)

        for channel_url in channels:
            channel = await client.get_entity(channel_url)
            result = await client(functions.channels.GetFullChannelRequest(channel))
            subscribers = result.full_chat.participants_count
            channel_name = channel.title

            print(f"Получение сообщений из канала {channel_name}...")
            i = 1
            async for message in client.iter_messages(channel, limit=None):
                if message.date < start_date or not message.text:
                    continue

                text = message.text.replace('\n', ' ').replace('\r', '')
                date_time = message.date.strftime("%Y-%m-%d %H:%M:%S")
                reactions = message.reactions

                messages.append([i, date_time, str(message.sender_id), channel_name, text, subscribers, str(reactions)])
                i += 1

    df = pd.DataFrame(messages,
                      columns=['№', 'Дата время', 'Источник', 'Название канала', 'Сообщение', 'Количество подписчиков',
                               'Реакции'])
    df.to_csv('tg_stock_messages.csv', index=False, encoding='utf-8')
    print(f"Сообщения из всех каналов сохранены в файл 'tg_stock_messages.csv'.")


async def main():
    await collect_all_tg_messages(tg_channels)


if __name__ == "__main__":
    asyncio.run(main())
