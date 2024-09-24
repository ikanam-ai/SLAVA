import re
import string
from typing import Final

# FILL THIS
DATA_EXPERIMENT_FOLDER: str = "qwen_temp"
EXPERIMENT_NAME: str = "qwen_temp"
# FILL THIS

REGEX = re.compile("[%s]" % re.escape(string.punctuation + "\\n"))

ALL_EXPERIMENTS_FOLDER: str = "experiments"
EXPERIMENT_FOLDER: str = f"{ALL_EXPERIMENTS_FOLDER}/{DATA_EXPERIMENT_FOLDER}"
FILE_NAME: str = (
    f"preprocessed/{DATA_EXPERIMENT_FOLDER}/preprocessed_{EXPERIMENT_NAME}.csv"
)

OPEN_QUESTION_TYPE_OF_DATA: str = "open_question"
NOT_OPEN_QUESTION_TYPE_OF_DATA: str = "not_open_question"

COLUMNS_FOR_PIVOT_TABLES: list[str] = [
    "Вид вопроса",
    "Предмет",
    "Провокационность оценка LLM 3 класса",
]

OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES: list[str] = [
    "exact_match",
    "levenshtein_ratio",
    "f1",
]
NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES: list[str] = [
    "exact_match",
    "is_substring",
    "partially_match",
]

# DataLoader
REPO_ID: Final[str] = "RANEPA-ai/SLAVA-OpenData-2800-v1"
REPO_TYPE: Final[str] = "dataset"

OPEN_DATASET_FILENAME: Final[str] = "open_questions_data.json"
