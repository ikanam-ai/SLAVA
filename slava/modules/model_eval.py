import logging
import os
import re
import time
from typing import Any

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

ERROR_COLUMN = "error"
ELAPSED_SEC_COLUMN = "elapsed_sec"


class ModelEval:
    def __init__(self):
        pass

    @staticmethod
    def _safe_filename(value: str, max_len: int = 180) -> str:
        value = str(value).replace("/", "-").replace("\\", "-").replace(":", "-")
        value = re.sub(r"[^0-9A-Za-zА-Яа-я._ -]+", "-", value)
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"-+", "-", value).strip("._- ")
        return value[:max_len] or "model"

    @staticmethod
    def _get_from_mapping(value: Any, key: str, default: Any = "") -> Any:
        return value.get(key, default) if isinstance(value, dict) else default

    @staticmethod
    def _extract_values(row: pd.Series):
        inputs = row.get(INPUTS_COLUMN, {})
        if not isinstance(inputs, dict):
            # Some prepared datasets already store the final prompt as a string.
            return {
                TASK_COLUMN: str(inputs),
                TEXT_COLUMN: "",
                **{f"Option_{i}": "" for i in range(1, 10)},
            }

        options = inputs.get(OPTIONS_COLUMN, {})
        if not isinstance(options, dict):
            options = {}

        values = {
            TASK_COLUMN: inputs.get(TASK_COLUMN, "") or "",
            TEXT_COLUMN: inputs.get(TEXT_COLUMN, "") or "",
        }

        for i in range(1, 10):
            option_key = OPTION_SUBCOLUMN_TEMPLATE.format(i)
            option_value = options.get(option_key, "")
            values[f"Option_{i}"] = "" if option_value is None else option_value

        return values

    def fill_instruction(self, row: pd.Series, prompt_instruction: str = PROMPT_INSTRUCTION) -> str:
        # Preferred format: instruction template + structured inputs dict.
        instruction_template = row.get(INSTRUCTION_COLUMN, "")
        inputs = row.get(INPUTS_COLUMN, "")

        if instruction_template:
            try:
                filled_instruction = str(instruction_template).format(**self._extract_values(row))
            except Exception as exc:
                logging.warning("Could not format instruction for id=%s: %r", row.get(ID_COLUMN, "N/A"), exc)
                filled_instruction = f"{instruction_template}\n{inputs}"
        else:
            filled_instruction = str(inputs)

        return f"{filled_instruction}{prompt_instruction}"

    @staticmethod
    def _meta_value(row: pd.Series, key: str, default: Any = "") -> Any:
        meta = row.get(META_COLUMN, {})
        if isinstance(meta, dict):
            return meta.get(key, default)
        return row.get(key, default)

    def run_evaluation(
        self,
        model_name: str,
        dataset: pd.DataFrame,
        model_handler: ModelHandler,
        folder_path: str = RESULTS_FILEPATH,
        max_retries: int = 3,
        base_delay_sec: float = 2.0,
    ) -> None:
        safe_model_name = self._safe_filename(model_name)
        results_filepath = os.path.join(folder_path, f"{safe_model_name}.csv")
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)

        results = []
        for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            prompt = self.fill_instruction(row)

            response_text = ""
            last_error = ""
            elapsed_sec = 0.0

            for attempt in range(1, max_retries + 1):
                started_at = time.time()
                try:
                    response = model_handler.generate_response(prompt)
                    elapsed_sec = time.time() - started_at
                    response_text = str(response or "").strip()
                    if not response_text:
                        raise ValueError("Empty model response")
                    last_error = ""
                    break
                except Exception as exc:
                    elapsed_sec = time.time() - started_at
                    last_error = repr(exc)
                    logging.warning(
                        "Error while calling model '%s' for id=%s (attempt %d/%d): %s",
                        model_name,
                        row.get(ID_COLUMN, "N/A"),
                        attempt,
                        max_retries,
                        last_error,
                    )
                    if attempt < max_retries:
                        time.sleep(base_delay_sec * attempt)

            if not response_text:
                response_text = f"[ERROR AFTER {max_retries} RETRIES]"

            results.append(
                {
                    ID_COLUMN: row.get(ID_COLUMN, ""),
                    MODEL_COLUMN: model_name,
                    SUBJECT_COLUMN: self._meta_value(row, SUBJECT_COLUMN),
                    TYPE_COLUMN: self._meta_value(row, TYPE_COLUMN),
                    PROVOC_SCORE_COLUMN: self._meta_value(row, PROVOC_SCORE_COLUMN),
                    INPUTS_COLUMN: prompt,
                    MODEL_ANSWER_COLUMN: response_text,
                    REAL_ANSWER_COLUMN: row.get(REAL_ANSWER_COLUMN, ""),
                    ERROR_COLUMN: last_error,
                    ELAPSED_SEC_COLUMN: round(elapsed_sec, 3),
                }
            )

            # Incremental save: long experiments can be resumed after SSH disconnects/crashes.
            pd.DataFrame(results).to_csv(results_filepath, index=False, encoding="utf-8-sig")

        logging.info("Results saved to %s", results_filepath)
