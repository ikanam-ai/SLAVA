import logging

import pandas as pd
from tqdm import tqdm

from slava.config import (
    INPUTS_COLUMN,
    INSTRUCTION_COLUMN,
    OPTION_SUBCOLUMN_TEMPLATE,
    OPTIONS_COLUMN,
    PROMPT_INSTRUCTION,
    RESULTS_FILEPATH,
    TASK_COLUMN,
    TEXT_COLUMN,
)
from slava.modules.model_handler import ModelHandler


class ModelEval:

    def __init__(
        self,
    ):
        self.results = []

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
        dataset: pd.DataFrame,
        model_handler: ModelHandler,
        results_filepath: str = RESULTS_FILEPATH,
    ) -> None:
        for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            prompt = self.fill_instruction(row)
            response = model_handler.generate_response(prompt)
            self.results.append(
                {
                    "id": row["id"],
                    "input": prompt,
                    "response": response.strip(),
                    "output": row["outputs"],
                }
            )

        pd.DataFrame(self.results).to_csv(results_filepath, index=False, encoding="utf-8")
        logging.info(f"Results saved to {results_filepath}")
