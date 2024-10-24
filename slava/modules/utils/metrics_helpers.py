from typing import List, Union

import pandas as pd
from fuzzywuzzy import fuzz

from slava.config import (
    AGGFUNC,
    EXACT_MATCH_COLUMN,
    F1_SCORE_COLUMN,
    IS_SUBSTRING_COLUMN,
    LEVENSHTEIN_RATIO_COLUMN,
    MODEL_ANSWER_COLUMN,
    MODEL_COLUMN,
    ONLY_NUMBERS_MODEL_ANSWER_COLUMN,
    PARTIALLY_MATCH_COLUMN,
    PROVOC_SCORE_COLUMN,
    REAL_ANSWER_COLUMN,
    SUBJECT_COLUMN,
    TYPE_COLUMN,
)
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
    ).astype(int)
    return not_open_questions


def partially_match(not_open_questions: pd.DataFrame) -> pd.DataFrame:
    not_open_questions[ONLY_NUMBERS_MODEL_ANSWER_COLUMN] = not_open_questions[MODEL_ANSWER_COLUMN].apply(only_numbers)
    not_open_questions[PARTIALLY_MATCH_COLUMN] = not_open_questions.apply(calculate_pm, axis=1)
    return not_open_questions


def create_pivot_table(
    questions_type: str,
    data: pd.DataFrame,
    value_columns: List[str],
    aggfunc: Union[str, List[str]] = AGGFUNC,
    fillna_value: float = 0,
) -> pd.DataFrame:
    pivot_by_question_type = data.pivot_table(
        values=value_columns, index=MODEL_COLUMN, columns=TYPE_COLUMN, aggfunc=aggfunc
    ).fillna(fillna_value)

    pivot_by_subject = data.pivot_table(
        values=value_columns, index=MODEL_COLUMN, columns=SUBJECT_COLUMN, aggfunc=aggfunc
    ).fillna(fillna_value)

    pivot_by_provocative_score = data.pivot_table(
        values=value_columns, index=MODEL_COLUMN, columns=PROVOC_SCORE_COLUMN, aggfunc=aggfunc
    ).fillna(fillna_value)

    combined_pivot = pd.concat(
        [pivot_by_question_type, pivot_by_subject, pivot_by_provocative_score],
        axis=1,
        keys=[TYPE_COLUMN, SUBJECT_COLUMN, PROVOC_SCORE_COLUMN],
    )

    combined_pivot.columns = [
        f"{questions_type}-{category}-{metric}-{level}" for category, level, metric in combined_pivot.columns
    ]

    return combined_pivot.reset_index()
