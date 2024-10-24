import pandas as pd

from slava.config import (
    AGGFUNC,
    COMBINED_VALUES_NAMING,
    KEYS_NAMING,
    LEADERBOARD_SHEET_NAME,
    METRICS_NAMING,
    MODEL_COLUMN,
    NOT_OPEN_QUESTION_TYPE_NAME,
    NOT_OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
    OPEN_QUESTION_TYPE_NAME,
    OPEN_QUESTION_VALUES_FOR_PIVOT_TABLES,
    QUESTION_TYPES_NAMING,
)
from slava.modules.utils.metrics_helpers import (
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

    def _get_metrics_table(self) -> pd.DataFrame:
        metrics_table_for_open_questions = self._create_metrics_table_for_open_questions()
        metrics_table_for_not_open_questions = self._create_metrics_table_for_not_open_questions()

        metrics_table = pd.concat([metrics_table_for_open_questions, metrics_table_for_not_open_questions], axis=1)
        metrics_table = metrics_table.loc[:, ~metrics_table.columns.duplicated()]
        metrics_table = metrics_table.round(2)
        return metrics_table

    def _get_renamed_metrics_table(
        self,
    ) -> pd.DataFrame:
        renamed_columns = []

        metrics_table = self._get_metrics_table()
        for column in metrics_table.columns:
            parts = column.split("-")
            if len(parts) == 4:
                questions_type_name = QUESTION_TYPES_NAMING.get(parts[0])
                pivot_column_name = KEYS_NAMING.get(parts[1])
                value_name = COMBINED_VALUES_NAMING.get(parts[2])
                metric_name = METRICS_NAMING.get(parts[3])
                renamed_name = f"{questions_type_name} {pivot_column_name} {value_name} {metric_name}".strip()
                renamed_columns.append(renamed_name)
            else:
                renamed_columns.append(column)

        metrics_table.columns = renamed_columns
        return metrics_table

    def save_metrics_table_to_excel(self, metrics_table_filename: str = "metrics_table.xlsx") -> None:
        renamed_metrics_table = self._get_renamed_metrics_table()
        leaderboard_values = []

        with pd.ExcelWriter(metrics_table_filename) as writer:
            for value in KEYS_NAMING.values():
                columns = [MODEL_COLUMN] + renamed_metrics_table.filter(like=value).columns.tolist()
                value_sheet = renamed_metrics_table[columns].copy()

                value_sheet.to_excel(writer, sheet_name=value, index=False)

                value_sheet[f"{value} {AGGFUNC}"] = value_sheet.drop(MODEL_COLUMN, axis=1).mean(axis=1)

                leaderboard_values.append(value_sheet[[MODEL_COLUMN, f"{value} {AGGFUNC}"]])

            leaderboard = leaderboard_values[0]
            for sheet_mean_data in leaderboard_values[1:]:
                leaderboard = leaderboard.merge(sheet_mean_data, on=MODEL_COLUMN, how="outer")

            leaderboard.to_excel(writer, sheet_name=LEADERBOARD_SHEET_NAME, index=False)
