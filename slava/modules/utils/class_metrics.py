import pandas as pd
from fuzzywuzzy import fuzz

from slava.config import *
from slava.modules.utils.metrics_utils import calculate_f1_score, calculate_pm, only_numbers


def exact_match(questions: pd.DataFrame) -> pd.DataFrame:
    questions[EXACT_MATCH_COLUMN] = (questions[REAL_ANSWER_COLUMN] == questions[MODEL_ANSWER_COLUMN]).astype(int)
    return questions


def levenshtein_ratio(open_questions: pd.DataFrame) -> pd.DataFrame:
    open_questions[LEVENSHTEIN_RATIO_COLUMN] = open_questions.apply(
        lambda row: fuzz.ratio(row[REAL_ANSWER_COLUMN], row[MODEL_ANSWER_COLUMN]) / 100, axis=1
    )
    return open_questions


def f1_score(open_questions: pd.DataFrame) -> pd.DataFrame:
    open_questions[F1_SCORE_COLUMN] = open_questions.apply(
        lambda row: calculate_f1_score(row[REAL_ANSWER_COLUMN], row[MODEL_ANSWER_COLUMN]), axis=1
    )
    return open_questions


def is_substring(not_open_questions: pd.DataFrame) -> pd.DataFrame:
    not_open_questions[IS_SUBSTRING_COLUMN] = not_open_questions.apply(
        lambda row: row[REAL_ANSWER_COLUMN] in row[MODEL_ANSWER_COLUMN], axis=1
    )
    return not_open_questions


def partially_match(not_open_questions: pd.DataFrame) -> pd.DataFrame:
    not_open_questions[ONLY_NUMBERS_MODEL_ANSWER_COLUMN] = not_open_questions[MODEL_ANSWER_COLUMN].apply(only_numbers)
    not_open_questions[PARTIALLY_MATCH_COLUMN] = not_open_questions.apply(calculate_pm, axis=1)
    return not_open_questions


def create_pivot_table(data: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    pivot_by_question_type = data.pivot_table(
        values=value_columns, index=INDEX_COLUMN, columns=TYPE_OF_QUESTION_COLUMN, aggfunc="mean"
    ).fillna(0)

    pivot_by_subject = data.pivot_table(
        values=value_columns, index=INDEX_COLUMN, columns=SUBJECT_COLUMN, aggfunc="mean"
    ).fillna(0)

    pivot_by_provocative_score = data.pivot_table(
        values=value_columns, index=INDEX_COLUMN, columns=PROVOCATIVE_SCORE_COLUMN, aggfunc="mean"
    ).fillna(0)

    combined_pivot = pd.concat(
        [pivot_by_question_type, pivot_by_subject, pivot_by_provocative_score],
        axis=1,
        keys=[TYPE_OF_QUESTION_COLUMN, SUBJECT_COLUMN, PROVOCATIVE_SCORE_COLUMN],
    )

    return combined_pivot
