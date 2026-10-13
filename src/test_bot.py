import os
import requests
from dotenv import load_dotenv

def main():
    load_dotenv()

    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    if not BOT_TOKEN or not CHAT_ID:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your environment.")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": "✂️ Bot test successful!"
    }

    response = requests.post(url, data=payload, timeout=10)

    response.raise_for_status()
    print("Telegram test completed.")


if __name__ == "__main__":
    main()
