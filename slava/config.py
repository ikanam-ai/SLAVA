from typing import Final

# Model eval
INPUTS_COLUMN: Final[str] = "inputs"
TASK_COLUMN: Final[str] = "task"
TEXT_COLUMN: Final[str] = "text"
INSTRUCTION_COLUMN: Final[str] = "instruction"
OPTIONS_COLUMNS: Final[str] = "options"
OPTION_SUBCOLUM_TEMPLATE: Final[str] = "option_{}"

# Metrics
OPEN_QUESTION_FLAG_COLUMN: Final[str] = "open_question_flag"
ONLY_NUMBERS_MODEL_ANSWER_COLUMN: Final[str] = "only_numbers_response"

OPEN_QUESTION_VALUE: Final[str] = "открытый ответ"
SINGLE_CHOICE: Final[str] = "выбор ответа (один)"
MULTI_CHOICE: Final[str] = "выбор ответа (мультивыбор)"
MATCHING: Final[str] = "установление соответствия"
SEQUENCE: Final[str] = "указание последовательности"

EXACT_MATCH_COLUMN: Final[str] = "exact_match"
LEVENSHTEIN_RATIO_COLUMN: Final[str] = "levenshtein_ratio"
F1_SCORE_COLUMN: Final[str] = "f1_score"
IS_SUBSTRING_COLUMN: Final[str] = "is_substring"
PARTIALLY_MATCH_COLUMN: Final[str] = "partially_match"

# Pivot tables
OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES: Final[list[str]] = [
    EXACT_MATCH_COLUMN,
    LEVENSHTEIN_RATIO_COLUMN,
    F1_SCORE_COLUMN,
]
NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES: Final[list[str]] = [
    EXACT_MATCH_COLUMN,
    IS_SUBSTRING_COLUMN,
    PARTIALLY_MATCH_COLUMN,
]

INDEX_COLUMN: Final = "model"
TYPE_OF_QUESTION_COLUMN: Final = "Вид вопроса"
SUBJECT_COLUMN: Final = "Предмет"
PROVOCATIVE_SCORE_COLUMN: Final = "Провокационность оценка LLM 3 класса"
REAL_ANSWER_COLUMN: Final[str] = "Ответ"
MODEL_ANSWER_COLUMN: Final[str] = "response"

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
