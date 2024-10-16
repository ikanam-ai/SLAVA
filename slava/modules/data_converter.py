import json

import pandas as pd

from slava.config import (
    COMMENT_COLUMN,
    ID_COLUMN,
    INPUTS_COLUMN,
    INSTRUCTION_COLUMN,
    META_COLUMN,
    OPTION_SUBCOLUMN_TEMPLATE,
    OPTIONS_COLUMN,
    PROVOC_SCORE_COLUMN,
    REAL_ANSWER_COLUMN,
    SOURCE_COLUMN,
    SUBJECT_COLUMN,
    TASK_COLUMN,
    TEXT_COLUMN,
    TYPE_COLUMN,
)


class DataConverter:

    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data
        self.json_objects: list = []

    def _create_json_objects(
        self,
    ):

        for _, row in self.data.iterrows():
            options = {
                OPTION_SUBCOLUMN_TEMPLATE.format(i): row[OPTION_SUBCOLUMN_TEMPLATE.format(i)] for i in range(1, 10)
            }

            json_object = {
                ID_COLUMN: row[ID_COLUMN],
                INSTRUCTION_COLUMN: row[INSTRUCTION_COLUMN],
                INPUTS_COLUMN: {TASK_COLUMN: row[TASK_COLUMN], TEXT_COLUMN: row[TEXT_COLUMN], OPTIONS_COLUMN: options},
                REAL_ANSWER_COLUMN: row[REAL_ANSWER_COLUMN],
                META_COLUMN: {
                    SUBJECT_COLUMN: row[SUBJECT_COLUMN],
                    TYPE_COLUMN: row[TYPE_COLUMN],
                    SOURCE_COLUMN: row[SOURCE_COLUMN],
                    COMMENT_COLUMN: row[COMMENT_COLUMN],
                    PROVOC_SCORE_COLUMN: row[PROVOC_SCORE_COLUMN],
                },
            }
            self.json_objects.append(json_object)

    def save_json_objects_to_jsonl(self, file_path: str = "open_questions_dataset.jsonl"):
        self._create_json_objects()

        with open(file_path, "w", encoding="utf-8") as file:
            for entry in self.json_objects:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
