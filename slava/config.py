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
FILE_NAME: str = f"preprocessed/{DATA_EXPERIMENT_FOLDER}/preprocessed_{EXPERIMENT_NAME}.csv"

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

MODEL_OUTPUT_COLUMN: Final[str] = "response"

INPUTS_COLUMN: Final[str] = "inputs"

TASK_COLUMN: Final[str] = "task"
TEXT_COLUMN: Final[str] = "text"

INSTRUCTION_COLUMN: Final[str] = "instruction"
OPTIONS_COLUMNS: Final[str] = "options"
OPTION_SUBCOLUM_TEMPLATE: Final[str] = "option_{}"

# DataLoader
REPO_ID: Final[str] = "RANEPA-ai/SLAVA-OpenData-2800-v1"
REPO_TYPE: Final[str] = "dataset"
OPEN_DATASET_FILENAME: Final[str] = "open_questions_data.jsonl"

REQUIRED_COLUMNS: Final[list[str]] = ["instruction", "inputs", "outputs", "meta"]

# Models
MODELS_TYPES: Final[tuple[str]] = ["gigachat", "huggingface", "ollama", "openai", "yandexgpt"]

# GigaChatModel
GIGACHAT_MODEL_SCOPE: Final[str] = "GIGACHAT_API_PERS"

# HuggingFaceModel
HUGGINGFACE_MODEL_MAX_TOKENS: Final[int] = 50
HUGGINGFACE_MODEL_TOP_K: Final[int] = 50
HUGGINGFACE_MODEL_TEMPERATURE: Final[float] = 1.0

# OllamaModel
OLLAMA_MODEL_NUM_CTX: Final[int] = 128
OLLAMA_MODEL_TEMPERATURE: Final[float] = 1.0
OLLAMA_MODEL_TOP_K: Final[int] = 50
OLLAMA_MODEL_MAX_TOKENS: Final[int] = 100

# OpenAIModel
OPENAI_MODEL_TEMPERATURE: Final[float] = 1.0

# YandexGPTModel
YANDEXGPT_URL: Final[str] = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEXGPT_MODEL_URI: Final[str] = "gpt://b1ggnkibfbsvf7qdohnd/yandexgpt"
YANDEXGPT_STREAM: Final[bool] = False
YANDEXGPT_TEMPERATURE: Final[float] = 0.0
YANDEXGPT_MAXTOKENS: Final[str] = "200"

# ModelEval
PROMPT_INSTRUCTION: Final[str] = (
    "\nСАМОЕ ВАЖНОЕ: Отвечай максимально кратко используя только цифры если они даны или слова в задачах с открытым ответом.\nОтвет: "
)
RESULTS_FILEPATH: Final[str] = "results.csv"
RESULTS_COLUMNS: Final[list[str]] = ["id", "input", "response", "output"]
