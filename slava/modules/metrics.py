import pandas as pd

from slava.config import (
    NOT_OPEN_QUESTION_TYPE_NAME,
    NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
    OPEN_QUESTION_TYPE_NAME,
    OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
)
from slava.modules.utils.class_metrics import (
    create_pivot_table,
    exact_match,
    f1_score,
    is_substring,
    levenshtein_ratio,
    partially_match,
)
from slava.modules.utils.metrics_utils import preprocess_answers


class MetricsCalculator:

    def __init__(self, data: pd.DataFrame):
        self.open_questions, self.not_open_questions = preprocess_answers(data)

        self.__calculate_metrics_for_open_questions()
        self.__calculate_metrics_for_not_open_questions()

    def __calculate_metrics_for_open_questions(self):
        self.open_questions = exact_match(self.open_questions)
        self.open_questions = levenshtein_ratio(self.open_questions)
        self.open_questions = f1_score(self.open_questions)

    def __calculate_metrics_for_not_open_questions(self):
        self.not_open_questions = exact_match(self.not_open_questions)
        self.not_open_questions = is_substring(self.not_open_questions)
        self.not_open_questions = partially_match(self.not_open_questions)

    def _create_metrics_table_for_open_questions(self):
        return create_pivot_table(
            questions_type=OPEN_QUESTION_TYPE_NAME,
            data=self.open_questions,
            value_columns=OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
        )

    def _create_metrics_table_for_not_open_questions(self):
        return create_pivot_table(
            questions_type=NOT_OPEN_QUESTION_TYPE_NAME,
            data=self.not_open_questions,
            value_columns=NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
        )

    def get_metrics_table(self) -> pd.DataFrame:
        metrics_table_for_open_questions = self._create_metrics_table_for_open_questions()
        metrics_table_for_not_open_questions = self._create_metrics_table_for_not_open_questions()

        combined_metrics = pd.concat([metrics_table_for_open_questions, metrics_table_for_not_open_questions], axis=1)
        return combined_metrics
