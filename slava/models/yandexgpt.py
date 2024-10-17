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
    def __init__(self, api_key: str = None, temperature: float = YANDEXGPT_TEMPERATURE):
        self.url = YANDEXGPT_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {api_key}",
        }
        self.temperature = temperature

    def get_response(self, prompt: str = None):
        completion_options = {
            "modelUri": YANDEXGPT_MODEL_URI,
            "completionOptions": {
                "stream": YANDEXGPT_STREAM,
                "temperature": self.temperature,
                "maxTokens": YANDEXGPT_MAXTOKENS,
            },
            "messages": [{"role": "user", "text": prompt}],
        }
        response = requests.post(self.url, headers=self.headers, json=completion_options)
        return response.text

    def process_dataframe(self, dataset: pd.DataFrame):
        res_list = []
        for instruction in tqdm(dataset[INSTRUCTION_COLUMN]):
            result = self.get_response(instruction)
            res_list.append(result)
            time.sleep(0.75)
        dataset[MODEL_ANSWER_COLUMN] = res_list
        return dataset
