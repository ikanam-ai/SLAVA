import logging
import os

import pandas as pd
from tqdm import tqdm

from slava.config import (
    ID_COLUMN,
    INPUTS_COLUMN,
    INSTRUCTION_COLUMN,
    META_COLUMN,
    MODEL_ANSWER_COLUMN,
    MODEL_COLUMN,
    OPTION_SUBCOLUMN_TEMPLATE,
    OPTIONS_COLUMN,
    PROMPT_INSTRUCTION,
    PROVOC_SCORE_COLUMN,
    REAL_ANSWER_COLUMN,
    RESULTS_FILEPATH,
    SUBJECT_COLUMN,
    TASK_COLUMN,
    TEXT_COLUMN,
    TYPE_COLUMN,
)
from slava.modules.model_handler import ModelHandler


class ModelEval:

    def __init__(
        self,
    ):
        pass

    @staticmethod
    def _extract_values(row: pd.Series):
        values = {
            TASK_COLUMN: row[INPUTS_COLUMN].get(TASK_COLUMN, ""),
            TEXT_COLUMN: row[INPUTS_COLUMN].get(TEXT_COLUMN, ""),
        }

        for i in range(1, 10):
            option_key = OPTION_SUBCOLUMN_TEMPLATE.format(i)
            values[f"Option_{i}"] = row[INPUTS_COLUMN][OPTIONS_COLUMN].get(option_key, "")

        values = {k: v for k, v in values.items() if pd.notna(v) and v != ""}

        return values

    def fill_instruction(self, row: pd.Series, prompt_instruction: str = PROMPT_INSTRUCTION) -> str:
        instruction_template = row[INSTRUCTION_COLUMN]

        values = self._extract_values(row)

        try:
            filled_instruction = instruction_template.format(**values) + prompt_instruction
        except KeyError as e:
            logging.info(f"Substitution error: missing key {e}")
            filled_instruction = "Ошибка: отсутствует необходимая информация для формирования запроса."

        return filled_instruction

    def run_evaluation(
        self,
        model_name: str,
        dataset: pd.DataFrame,
        model_handler: ModelHandler,
        folder_path: str = RESULTS_FILEPATH,
    ) -> None:
        safe_model_name = model_name.replace("/", "-")
        results_filepath = os.path.join(folder_path, f"{safe_model_name}.csv")
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)

        results = []
        for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            prompt = self.fill_instruction(row)
            response = model_handler.generate_response(prompt)
            results.append(
                {
                    ID_COLUMN: row[ID_COLUMN],
                    MODEL_COLUMN: model_name,
                    SUBJECT_COLUMN: row[META_COLUMN][SUBJECT_COLUMN],
                    TYPE_COLUMN: row[META_COLUMN][TYPE_COLUMN],
                    PROVOC_SCORE_COLUMN: row[META_COLUMN][PROVOC_SCORE_COLUMN],
                    INPUTS_COLUMN: prompt,
                    MODEL_ANSWER_COLUMN: response.strip(),
                    REAL_ANSWER_COLUMN: row[REAL_ANSWER_COLUMN],
                }
            )

        pd.DataFrame(results).to_csv(results_filepath, index=False, encoding="utf-8")
        logging.info(f"Results saved to {results_filepath}")
