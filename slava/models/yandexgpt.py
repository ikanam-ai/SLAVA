import time

import pandas as pd
import requests
from tqdm import tqdm

from slava.config import (
    INSTRUCTION_COLUMN,
    MODEL_ANSWER_COLUMN,
    YANDEXGPT_MAXTOKENS,
    YANDEXGPT_MODEL_URI,
    YANDEXGPT_STREAM,
    YANDEXGPT_TEMPERATURE,
    YANDEXGPT_URL,
)


class YandexGPTModel:
    def __init__(
        self,
        uri: str = None,
        api_key: str = None,
        temperature: float = YANDEXGPT_TEMPERATURE,
        delay_sec: float = 0.75,  # задержка между запросами
    ):
        # uri — полный modelUri вида "gpt://<folder-id>/yandexgpt-lite/latest"
        self.uri = uri or YANDEXGPT_MODEL_URI
        self.url = YANDEXGPT_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {api_key}",
        }
        self.temperature = temperature
        self.delay_sec = delay_sec

    def get_response(self, prompt: str = None) -> str:
        completion_options = {
            "modelUri": self.uri,
            "completionOptions": {
                "stream": YANDEXGPT_STREAM,
                "temperature": self.temperature,
                "maxTokens": YANDEXGPT_MAXTOKENS,
            },
            "messages": [{"role": "user", "text": prompt}],
        }

        response = requests.post(self.url, headers=self.headers, json=completion_options)
        response.raise_for_status()  # если 4xx/5xx — сразу бросаем исключение

        data = response.json()

        try:
            text = data["result"]["alternatives"][0]["message"]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected YandexGPT response structure: {data}") from e

        # Задержка между запросами — "безопасный" rate limit
        time.sleep(self.delay_sec)

        return text

    def process_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        res_list = []
        for instruction in tqdm(dataset[INSTRUCTION_COLUMN]):
            result = self.get_response(instruction)  # внутри уже есть задержка
            res_list.append(result)
        dataset[MODEL_ANSWER_COLUMN] = res_list
        return dataset
