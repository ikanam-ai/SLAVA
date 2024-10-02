import pandas as pd

from slava.config import *
from slava.modules.utils.class_metrics import *
from slava.modules.utils.metrics_utils import *


class MetricsCalculator:

    def __init__(self, data: pd.DataFrame):
        self.open_questions, self.not_open_questions = preprocess_answers(data)

        self.__calculate_metrics_for_open_questions()
        self.__calculate_metrics_for_not_open_questions()

    def __calculate_metrics_for_open_questions(
        self,
    ):
        self.open_questions = exact_match(self.open_questions)
        self.open_questions = levenshtein_ratio(self.open_questions)
        self.open_questions = f1_score(self.open_questions)

    def __calculate_metrics_for_not_open_questions(
        self,
    ):
        self.not_open_questions = exact_match(self.not_open_questions)
        self.not_open_questions = is_substring(self.not_open_questions)
        self.not_open_questions = partially_match(self.not_open_questions)

    def create_pivot_table_for_open_questions(
        self,
    ):
        return create_pivot_table(self.open_questions, OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES)

    def create_pivot_table_for_not_open_questions(
        self,
    ):
        return create_pivot_table(self.not_open_questions, NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES)
