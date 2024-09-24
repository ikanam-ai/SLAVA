"""Module for getting metrics of model responses for a benchmark."""

import collections
import os
import re
import string
import warnings

import pandas as pd
from fuzzywuzzy import fuzz

from slava.config import *

warnings.filterwarnings("ignore")


def read_file(file_path: str) -> pd.DataFrame:
    file_extension = file_path.split(".")[-1].lower()

    if file_extension == "csv":
        return pd.read_csv(file_path)
    elif file_extension == "xlsx":
        return pd.read_excel(file_path)
    else:
        raise ValueError(
            "Unsupported file extension. Please provide a '.csv' or '.xlsx' file."
        )


def only_numbers(raw_text: str) -> str:
    """
    Extracts the first sequence of digits from the provided text.

    Parameters:
    raw_text (str): The raw text from which to extract digits.

    Returns:
    str: A string consisting of the first sequence of digits found or "-" if no digits are found.
    """
    # Try to extract the first paragraph if possible
    try:
        text = re.search(".*\\n", raw_text).group(0)  # Using .group(0) to get the match
    except AttributeError:
        text = raw_text  # Use the entire text if no newlines are found

    # Remove punctuation using a predefined REGEX pattern
    text = REGEX.sub("", text)

    # Join text without spaces to form a continuous string
    text = "".join(text.split())

    # Find the first sequence of digits
    match = re.search(r"\d+", text)
    if match:
        return match.group(0)  # Return the first sequence of digits
    else:
        return "-"  # Return "-" if no digits are found


def answers_preprocessing(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data["response"] = data["response"].astype(str).str.lower().str.strip()
    data["Ответ"] = data["Ответ"].astype(str).str.lower().str.strip()

    return data


def split_by_type_question(
    data: pd.DataFrame,
    type_question_column: str = "Вид вопроса",
    type_open_question: str = "открытый ответ",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_question = data.loc[data[type_question_column] == type_open_question]
    not_open_question = data.loc[data[type_question_column] != type_open_question]

    return open_question, not_open_question


def exact_match(
    data: pd.DataFrame,
    real_answer_column: str = "Ответ",
    model_answer_column: str = "response",
) -> pd.DataFrame:
    data["exact_match"] = data[real_answer_column] == data[model_answer_column]

    return data


def is_substring(
    data: pd.DataFrame,
    real_answer_column: str = "Ответ",
    model_answer_column: str = "response",
) -> pd.DataFrame:
    data["is_substring"] = data.apply(
        lambda row: row[real_answer_column] in row[model_answer_column], axis=1
    )

    return data


def calculate_pm(
    row: pd.Series,
    real_answer_column: str = "Ответ",
    model_answer_column: str = "response",
    type_question_column: str = "Вид вопроса",
) -> float:
    def compute_one_choice(answer: str, model_answer: str) -> float:
        return 1.0 if answer == model_answer else 0.0

    def compute_multi_choice(answer: str, model_answer: str) -> float:
        if answer == model_answer:
            return 1.0
        if len(answer) == len(model_answer):
            differences = sum(a != b for a, b in zip(answer, model_answer))
            if differences == 1:
                return 0.5
        if abs(len(answer) - len(model_answer)) == 1:
            return 0.5
        return 0.0

    def compute_matching(answer: str, model_answer: str) -> float:
        if answer == model_answer:
            return 1.0
        if len(answer) == len(model_answer):
            differences = sum(a != b for a, b in zip(answer, model_answer))
            if differences == 1:
                return 0.5
        return 0.0

    def compute_sequence(answer: str, model_answer: str) -> float:
        if answer == model_answer:
            return 1.0
        if len(answer) == len(model_answer):
            differences = sum(a != b for a, b in zip(answer, model_answer))
            if differences == 1:
                return 0.5
        return 0.0

    question_type = row[type_question_column]
    answer = row[real_answer_column]
    model_answer = row[model_answer_column]

    if question_type == "выбор ответа (один)":
        return compute_one_choice(answer, model_answer)

    elif question_type == "выбор ответа (мультивыбор)":
        return compute_multi_choice(answer, model_answer)

    elif question_type == "установление соответствия":
        return compute_matching(answer, model_answer)

    elif question_type == "указание последовательности":
        return compute_sequence(answer, model_answer)

    else:
        raise ValueError("Неизвестный тип вопроса")


def partially_match(
    data,
):
    data["partially_match"] = data.apply(calculate_pm, axis=1)
    return data


def levenshtein_ratio(row: pd.Series) -> int:
    return fuzz.ratio(row["response"], row["Ответ"])


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_metrics_open_question(open_question: pd.DataFrame) -> None:
    open_question = exact_match(open_question)

    open_question["levenshtein_ratio"] = open_question.apply(levenshtein_ratio, axis=1)

    open_question["f1"] = open_question.apply(
        lambda row: compute_f1(row["Ответ"], row["response"]), axis=1
    )

    return open_question


def get_metrics_not_open_question(
    not_open_question: pd.DataFrame,
    model_answer_column: str = "response",
) -> None:
    not_open_question = exact_match(not_open_question)

    not_open_question = is_substring(not_open_question)

    not_open_question[model_answer_column] = not_open_question[
        model_answer_column
    ].apply(only_numbers)

    not_open_question = partially_match(not_open_question)

    return not_open_question


def create_and_save_pivot_table(
    type_of_data: str,
    data: pd.DataFrame,
    columns: str,
    values: list[str],
    index_column: str = "model",
    aggfunc: str = "mean",
    metrics_naming: str = "Метрика",
) -> None:
    pivot_table = pd.pivot_table(
        data, index=index_column, columns=columns, values=values, aggfunc=aggfunc
    )

    pivot_table = pivot_table.swaplevel(axis=1).sort_index(axis=1)
    pivot_table.columns.names = [columns, metrics_naming]

    pivot_table.to_csv(
        f"{EXPERIMENT_FOLDER}/{EXPERIMENT_NAME}_{type_of_data}_pivot_table_by_{columns}.csv"
    )


def run() -> None:
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

    data = read_file(file_path=FILE_NAME)

    data = answers_preprocessing(data)

    open_question, not_open_question = split_by_type_question(data)

    if EXPERIMENT_NAME == "random":
        not_open_question = get_metrics_not_open_question(not_open_question)

        for column in COLUMNS_FOR_PIVOT_TABLES:
            create_and_save_pivot_table(
                NOT_OPEN_QUESTION_TYPE_OF_DATA,
                not_open_question,
                column,
                NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
            )
    else:
        open_question = get_metrics_open_question(open_question)
        not_open_question = get_metrics_not_open_question(not_open_question)

        for column in COLUMNS_FOR_PIVOT_TABLES:
            create_and_save_pivot_table(
                OPEN_QUESTION_TYPE_OF_DATA,
                open_question,
                column,
                OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
            )

            create_and_save_pivot_table(
                NOT_OPEN_QUESTION_TYPE_OF_DATA,
                not_open_question,
                column,
                NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
            )


if __name__ == "__main__":
    run()
