import collections
import re
import string

import pandas as pd

from slava.config import (
    MATCHING,
    MODEL_ANSWER_COLUMN,
    MULTI_CHOICE,
    ONLY_NUMBERS_MODEL_ANSWER_COLUMN,
    OPEN_QUESTION_FLAG_COLUMN,
    OPEN_QUESTION_VALUE,
    REAL_ANSWER_COLUMN,
    SEQUENCE,
    SINGLE_CHOICE,
    TYPE_COLUMN,
)

# Utils
REGEX = re.compile("[%s]" % re.escape(string.punctuation + "\\n"))


def preprocess_answers(data: pd.DataFrame):
    data[REAL_ANSWER_COLUMN] = data[REAL_ANSWER_COLUMN].astype(str).str.lower().str.strip()
    data[MODEL_ANSWER_COLUMN] = data[MODEL_ANSWER_COLUMN].astype(str).str.lower().str.strip()

    data[OPEN_QUESTION_FLAG_COLUMN] = (data[TYPE_COLUMN] == OPEN_QUESTION_VALUE).astype(int)

    open_questions = data[data[OPEN_QUESTION_FLAG_COLUMN] == 1].reset_index(drop=True)
    not_open_questions = data[data[OPEN_QUESTION_FLAG_COLUMN] == 0].reset_index(drop=True)

    return open_questions, not_open_questions


def only_numbers(raw_text: str) -> str:
    try:
        text = re.search(".*\\n", raw_text).group(0)
    except AttributeError:
        text = raw_text

    text = REGEX.sub("", text)
    text = "".join(text.split())

    match = re.search(r"\d+", text)
    if match:
        return match.group(0)
    else:
        return "-"


def compute_one_choice(answer: str, only_numbers_model_answer: str) -> float:
    return 1.0 if answer == only_numbers_model_answer else 0.0


def compute_multi_choice(answer: str, only_numbers_model_answer: str) -> float:
    if answer == only_numbers_model_answer:
        return 1.0
    if len(answer) == len(only_numbers_model_answer):
        differences = sum(a != b for a, b in zip(answer, only_numbers_model_answer))
        if differences == 1:
            return 0.5
    if abs(len(answer) - len(only_numbers_model_answer)) == 1:
        return 0.5
    return 0.0


def compute_matching(answer: str, only_numbers_model_answer: str) -> float:
    if answer == only_numbers_model_answer:
        return 1.0
    if len(answer) == len(only_numbers_model_answer):
        differences = sum(a != b for a, b in zip(answer, only_numbers_model_answer))
        if differences == 1:
            return 0.5
    return 0.0


def compute_sequence(answer: str, only_numbers_model_answer: str) -> float:
    if answer == only_numbers_model_answer:
        return 1.0
    if len(answer) == len(only_numbers_model_answer):
        differences = sum(a != b for a, b in zip(answer, only_numbers_model_answer))
        if differences == 1:
            return 0.5
    return 0.0


def get_match_function(question_type: str):
    match_functions = {
        SINGLE_CHOICE: compute_one_choice,
        MULTI_CHOICE: compute_multi_choice,
        MATCHING: compute_matching,
        SEQUENCE: compute_sequence,
    }

    if question_type not in match_functions:
        raise ValueError("Unknown type of question")

    return match_functions[question_type]


def calculate_pm(
    row: pd.Series,
) -> float:

    question_type = row[TYPE_COLUMN]
    answer = row[REAL_ANSWER_COLUMN]
    only_numbers_model_answer = row[ONLY_NUMBERS_MODEL_ANSWER_COLUMN]

    match_function = get_match_function(question_type)
    return match_function(answer, only_numbers_model_answer)


def calculate_f1_score(real_answer, model_answer):
    real_answer_tokens = get_tokens(real_answer)
    model_answer_tokens = get_tokens(model_answer)

    common = collections.Counter(real_answer_tokens) & collections.Counter(model_answer_tokens)
    num_same = sum(common.values())

    if len(real_answer_tokens) == 0 or len(model_answer_tokens) == 0:
        return int(real_answer_tokens == model_answer_tokens)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(model_answer_tokens)
    recall = 1.0 * num_same / len(real_answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punctuation(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def to_lower(text: str) -> str:
    return text.lower()


def normalize_answer(text: str) -> str:
    lowered = to_lower(text)
    without_punctuation = remove_punctuation(lowered)
    without_articles = remove_articles(without_punctuation)
    normalized = white_space_fix(without_articles)

    return normalized


def get_tokens(text):
    if not text:
        return []
    return normalize_answer(text).split()
