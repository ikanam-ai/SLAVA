from typing import Final

# Questions Dataset
QUESTIONS_DATASET_COLUMNS: Final[list[str]] = [
    "id",  # Unique identifier for each question entry.
    "subject",  # The subject or topic to which the question belongs.
    "type",  # The type of question (e.g., "открытый ответ").
    "instruction",  # Instructions on how to answer the question.
    "task",  # The specific task or action required from the respondent.
    "text",  # The main content of the question or prompt.
    "option_1",  # The first option for multiple-choice answers.
    "option_2",  # The second option for multiple-choice answers.
    "option_3",  # The third option for multiple-choice answers.
    "option_4",  # The fourth option for multiple-choice answers.
    "option_5",  # The fifth option for multiple-choice answers.
    "option_6",  # The sixth option for multiple-choice answers.
    "option_7",  # The seventh option for multiple-choice answers.
    "option_8",  # The eighth option for multiple-choice answers.
    "option_9",  # The ninth option for multiple-choice answers.
    "outputs",  # The expected or correct output for the question.
    "source",  # The source from which the question was derived.
    "comment",  # Any additional comments or notes related to the question.
    "provoc_score",  # The provocation score associated with the question, often used for evaluation.
]

# Columns
INSTRUCTION_COLUMN: Final[str] = "instruction"
INPUTS_COLUMN: Final[str] = "inputs"
TASK_COLUMN: Final[str] = "task"
TEXT_COLUMN: Final[str] = "text"
OPTIONS_COLUMN: Final[str] = "options"
OPTION_SUBCOLUMN_TEMPLATE: Final[str] = "option_{}"
META_COLUMN: Final[str] = "meta"
ID_COLUMN: Final[str] = "id"
SUBJECT_COLUMN: Final[str] = "subject"
TYPE_COLUMN: Final[str] = "type"
SOURCE_COLUMN: Final[str] = "source"
COMMENT_COLUMN: Final[str] = "comment"
PROVOC_SCORE_COLUMN: Final[str] = "provoc_score"

# Metrics
MODEL_COLUMN: Final = "model"
REAL_ANSWER_COLUMN: Final[str] = "outputs"
MODEL_ANSWER_COLUMN: Final[str] = "response"

OPEN_QUESTION_FLAG_COLUMN: Final[str] = "open_question_flag"
ONLY_NUMBERS_MODEL_ANSWER_COLUMN: Final[str] = "only_numbers_response"

OPEN_QUESTION_TYPE_NAME: Final[str] = "open_question"
NOT_OPEN_QUESTION_TYPE_NAME: Final[str] = "not_open_question"

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

QUESTION_TYPES_NAMING = {OPEN_QUESTION_TYPE_NAME: "Open", NOT_OPEN_QUESTION_TYPE_NAME: "Not open"}

LEADERBOARD_SHEET_NAME = "Leaderboard"
TYPE_OF_QUESTION_SHEET_NAME = "Type of question"
SUBJECT_SHEET_NAME = "Subject"
PROVOCATIVENESS_SHEET_NAME = "Provocativeness"

KEYS_NAMING = {
    TYPE_COLUMN: TYPE_OF_QUESTION_SHEET_NAME,
    SUBJECT_COLUMN: SUBJECT_SHEET_NAME,
    PROVOC_SCORE_COLUMN: PROVOCATIVENESS_SHEET_NAME,
}

SUBJECTS_NAMING = {
    "Обществознание": "Social studies",
    "История": "History",
    "География": "Geography",
    "Политология": "Political science",
}

QUESTION_VALUES_NAMING = {
    OPEN_QUESTION_VALUE: "Open answer",
    SINGLE_CHOICE: "Single choice",
    MULTI_CHOICE: "Multiple choice",
    MATCHING: "Matching",
    SEQUENCE: "Sequence",
}

PROVOCATIVENESS_NAMING = {"1": "Low", "2": "Medium", "3": "High"}

COMBINED_VALUES_NAMING = {**SUBJECTS_NAMING, **QUESTION_VALUES_NAMING, **PROVOCATIVENESS_NAMING}

METRICS_NAMING = {
    EXACT_MATCH_COLUMN: "EM",
    LEVENSHTEIN_RATIO_COLUMN: "LR",
    F1_SCORE_COLUMN: "F1",
    IS_SUBSTRING_COLUMN: "IS",
    PARTIALLY_MATCH_COLUMN: "PM",
}

AGGFUNC: Final[str] = "mean"

# DataLoader
REPO_ID: Final[str] = "RANEPA-ai/SLAVA-OpenData-2800-v1"
REPO_TYPE: Final[str] = "dataset"
OPEN_DATASET_FILENAME: Final[str] = "open_questions_dataset.jsonl"
REQUIRED_COLUMNS: Final[list[str]] = [INSTRUCTION_COLUMN, INPUTS_COLUMN, REAL_ANSWER_COLUMN, META_COLUMN]

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

# Models
MODELS_TYPES: Final[tuple[str]] = ["gigachat", "huggingface", "ollama", "openai", "yandexgpt"]
DEVICE: Final[int] = 0

# ClaudeModel
CLAUDE_MODEL_NAME: Final[str] = "claude-3-5-sonnet-20240620"
CLAUDE_MODEL_MAX_TOKENS: Final[int] = 150

# GeminiModel
GEMINI_MODEL_NAME: Final[str] = "gemini-1.5-flash"

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
OPENAI_MODEL_NAME: Final[str] = "gpt-4o"
OPENAI_MODEL_MAX_TOKENS: Final[int] = 150
OPENAI_MODEL_TEMPERATURE: Final[float] = 0.7

# YandexGPTModel
YANDEXGPT_URL: Final[str] = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEXGPT_MODEL_URI: Final[str] = "gpt://{}/yandexgpt"
YANDEXGPT_STREAM: Final[bool] = False
YANDEXGPT_TEMPERATURE: Final[float] = 0.0
YANDEXGPT_MAXTOKENS: Final[str] = "200"

# ModelEval
PROMPT_INSTRUCTION: Final[str] = (
    "\nСАМОЕ ВАЖНОЕ: Отвечай максимально кратко используя только цифры если они даны или слова в задачах с открытым ответом.\nОтвет: "
)
RESULTS_FILEPATH: Final[str] = "results.csv"
