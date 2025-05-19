import pandas as pd

from slava.config import *
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

        for col in metrics_table.columns:
            if col != "model" and pd.api.types.is_numeric_dtype(metrics_table[col]):
                metrics_table[col] = metrics_table[col] * 100

        metrics_table = metrics_table.round(0)
        return metrics_table

    def _filter_existing(self, metrics_table: pd.DataFrame, columns: list[str]) -> List[str]:
        return [column for column in columns if column in metrics_table.columns]

    def _build_group(
        self, metrics_table: pd.DataFrame, groups: dict[str, list[str]], suffix: str = "avg"
    ) -> pd.DataFrame:
        parts: List[pd.DataFrame | pd.Series] = []
        present_columns: List[str] = []

        for name, cols in groups.items():
            real_columns = self._filter_existing(metrics_table, cols)

            if not real_columns:
                continue

            parts.append(metrics_table[real_columns])
            parts.append(metrics_table[real_columns].mean(axis=1).rename(f"{name}_{suffix}"))
            present_columns.extend(real_columns)

        if present_columns:
            parts.append(metrics_table[present_columns].mean(axis=1).rename(f"ALL_{suffix}"))

        return pd.concat(parts, axis=1) if parts else pd.DataFrame()

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

    def save_metrics_table_of_custom_dataset_to_excel(
        self, metrics_table_filename: str = "metrics_table_of_custom_dataset.xlsx"
    ) -> None:
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

        print(f"Excel-файл '{metrics_table_filename}' успешно создан.")

    def save_metrics_table_to_excel(self, metrics_table_filename: str = "metrics_table.xlsx") -> None:
        """
        Создаёт Excel-файл с четырьмя вкладками:
        - Subject
        - Type of question
        - Provocativeness
        - Leaderboard
        """
        renamed_metrics_table = self._get_renamed_metrics_table()

        renamed_metrics_table = renamed_metrics_table.rename(columns=COLUMN_RENAME_MAP)
        # ---------------------------
        # Вкладка "Subject"
        # ---------------------------
        # Вычисляем средние по группам
        geo_avg = renamed_metrics_table[GEO_COLS].mean(axis=1).rename("GEO_avg")
        history_avg = renamed_metrics_table[HISTORY_COLS].mean(axis=1).rename("HIST_avg")
        social_avg = renamed_metrics_table[SOCIAL_COLS].mean(axis=1).rename("SOC_avg")
        pol_avg = renamed_metrics_table[POL_COLS].mean(axis=1).rename("POL_avg")

        # DOMAIN_avg – среднее по всем выбранным колонкам
        domain_cols = GEO_COLS + HISTORY_COLS + SOCIAL_COLS + POL_COLS
        domain_avg = renamed_metrics_table[domain_cols].mean(axis=1).rename("DOMAIN_avg")

        # Итоговый DataFrame для вкладки Subject
        df_subject = pd.concat(
            [
                renamed_metrics_table[GEO_COLS],
                geo_avg,
                renamed_metrics_table[HISTORY_COLS],
                history_avg,
                renamed_metrics_table[SOCIAL_COLS],
                social_avg,
                renamed_metrics_table[POL_COLS],
                pol_avg,
                domain_avg,
            ],
            axis=1,
        )

        # Добавляем колонку "model"
        df_subject = pd.concat([renamed_metrics_table[["model"]], df_subject], axis=1)

        df_subject.to_csv("df_subject.csv")

        # ---------------------------
        # Вкладка "Type of question"
        # ---------------------------
        multich_avg = renamed_metrics_table[MULTICH_COLS].mean(axis=1).rename("NUM_Q_multich_avg")
        onech_avg = renamed_metrics_table[ONECH_COLS].mean(axis=1).rename("NUM_Q_onech_avg")
        seq_avg = renamed_metrics_table[SEQ_COLS].mean(axis=1).rename("NUM_Q_seq_avg")
        map_avg = renamed_metrics_table[MAP_COLS].mean(axis=1).rename("NUM_Q_map_avg")
        open_avg = renamed_metrics_table[OPEN_COLS].mean(axis=1).rename("OPEN_Q_avg")

        # Q_TYPE_avg – среднее по всем метрикам во вкладке
        qtype_cols = MULTICH_COLS + ONECH_COLS + SEQ_COLS + MAP_COLS + OPEN_COLS
        qtype_avg = renamed_metrics_table[qtype_cols].mean(axis=1).rename("Q_TYPE_avg")

        df_qtype = pd.concat(
            [
                renamed_metrics_table[MULTICH_COLS],
                multich_avg,
                renamed_metrics_table[ONECH_COLS],
                onech_avg,
                renamed_metrics_table[SEQ_COLS],
                seq_avg,
                renamed_metrics_table[MAP_COLS],
                map_avg,
                renamed_metrics_table[OPEN_COLS],
                open_avg,
                qtype_avg,
            ],
            axis=1,
        )

        # Добавляем колонку "model"
        df_qtype = pd.concat([renamed_metrics_table[["model"]], df_qtype], axis=1)

        # ---------------------------
        # Вкладка "Provocativeness"
        # ---------------------------
        provoc_low_avg = renamed_metrics_table[PROVOC_LOW_COLS].mean(axis=1).rename("PROVOC_1_avg")
        provoc_med_avg = renamed_metrics_table[PROVOC_MED_COLS].mean(axis=1).rename("PROVOC_2_avg")
        provoc_high_avg = renamed_metrics_table[PROVOC_HIGH_COLS].mean(axis=1).rename("PROVOC_3_avg")

        provoc_all_cols = PROVOC_LOW_COLS + PROVOC_MED_COLS + PROVOC_HIGH_COLS
        provoc_avg = renamed_metrics_table[provoc_all_cols].mean(axis=1).rename("PROVOC_avg")

        df_provoc = pd.concat(
            [
                renamed_metrics_table[PROVOC_LOW_COLS],
                provoc_low_avg,
                renamed_metrics_table[PROVOC_MED_COLS],
                provoc_med_avg,
                renamed_metrics_table[PROVOC_HIGH_COLS],
                provoc_high_avg,
                provoc_avg,
            ],
            axis=1,
        )

        # Добавляем колонку "model"
        df_provoc = pd.concat([renamed_metrics_table[["model"]], df_provoc], axis=1)

        # ---------------------------
        # Вкладка "Leaderboard"
        # ---------------------------
        df_leaderboard = pd.DataFrame(
            {
                "model": renamed_metrics_table["model"],
                "DOMAIN_avg": domain_avg,
                "Q_TYPE_avg": qtype_avg,
                "PROVOC_avg": provoc_avg,
            }
        )

        # Рассчитываем ALL_avg через pandas (среднее по столбцам DOMAIN_avg, Q_TYPE_avg, PROVOC_avg)
        df_leaderboard["ALL_avg"] = df_leaderboard[["DOMAIN_avg", "Q_TYPE_avg", "PROVOC_avg"]].mean(axis=1)

        # ---------------------------
        # Запись в Excel
        # ---------------------------
        with pd.ExcelWriter(metrics_table_filename) as writer:
            df_subject.to_excel(writer, sheet_name="Subject", index=False)
            df_qtype.to_excel(writer, sheet_name="Type of question", index=False)
            df_provoc.to_excel(writer, sheet_name="Provocativeness", index=False)
            df_leaderboard.to_excel(writer, sheet_name="Leaderboard", index=False)

        print(f"Excel-файл '{metrics_table_filename}' успешно создан.")

    def save_metrics_table_to_excel_dynamic(self, metrics_table_filename: str = "metrics_table.xlsx") -> None:
        """Гибко формирует Excel‑файл с вкладками *Subject*, *Type of question*, *Provocativeness*, *Leaderboard*.

        Функция не ломается, если каких‑то групп колонок нет (например, только «История» и «География»).
        В таблицы попадают только реально существующие колонки и агрегаты.
        """

        # --- Подготовка данных --------------------------------------------------
        renamed_metrics_table = self._get_renamed_metrics_table().rename(columns=COLUMN_RENAME_MAP)

        # --- Subject ------------------------------------------------------------
        domain_groups = {
            "GEO": GEO_COLS,
            "HIST": HISTORY_COLS,
            "SOC": SOCIAL_COLS,
            "POL": POL_COLS,
        }
        df_subject_body = self._build_group(renamed_metrics_table, domain_groups, suffix="avg")
        df_subject = (
            pd.concat([renamed_metrics_table[["model"]], df_subject_body], axis=1)
            if not df_subject_body.empty
            else pd.DataFrame()
        )

        # --- Type of question ---------------------------------------------------
        qtype_groups = {
            "NUM_Q_multich": MULTICH_COLS,
            "NUM_Q_onech": ONECH_COLS,
            "NUM_Q_seq": SEQ_COLS,
            "NUM_Q_map": MAP_COLS,
            "OPEN_Q": OPEN_COLS,
        }
        df_qtype_body = self._build_group(renamed_metrics_table, qtype_groups, suffix="avg")
        df_qtype = (
            pd.concat([renamed_metrics_table[["model"]], df_qtype_body], axis=1)
            if not df_qtype_body.empty
            else pd.DataFrame()
        )

        # --- Provocativeness ----------------------------------------------------
        provoc_groups = {
            "PROVOC_1": PROVOC_LOW_COLS,
            "PROVOC_2": PROVOC_MED_COLS,
            "PROVOC_3": PROVOC_HIGH_COLS,
        }
        df_provoc_body = self._build_group(renamed_metrics_table, provoc_groups, suffix="avg")
        df_provoc = (
            pd.concat([renamed_metrics_table[["model"]], df_provoc_body], axis=1)
            if not df_provoc_body.empty
            else pd.DataFrame()
        )

        # --- Leaderboard --------------------------------------------------------
        # Берём агрегаты, которые действительно существуют
        leaderboard_cols = {}
        if "ALL_avg" in df_subject_body.columns:
            leaderboard_cols["DOMAIN_avg"] = df_subject_body["ALL_avg"]
        if "ALL_avg" in df_qtype_body.columns:
            leaderboard_cols["Q_TYPE_avg"] = df_qtype_body["ALL_avg"]
        if "ALL_avg" in df_provoc_body.columns:
            leaderboard_cols["PROVOC_avg"] = df_provoc_body["ALL_avg"]

        if leaderboard_cols:
            df_leaderboard = pd.concat([renamed_metrics_table[["model"]], pd.DataFrame(leaderboard_cols)], axis=1)
            df_leaderboard["ALL_avg"] = df_leaderboard.drop(columns=["model"]).mean(axis=1)
        else:
            df_leaderboard = pd.DataFrame()

        # --- Запись в Excel -----------------------------------------------------
        with pd.ExcelWriter(metrics_table_filename) as writer:
            if not df_subject.empty:
                df_subject.to_excel(writer, sheet_name="Subject", index=False)
            if not df_qtype.empty:
                df_qtype.to_excel(writer, sheet_name="Type of question", index=False)
            if not df_provoc.empty:
                df_provoc.to_excel(writer, sheet_name="Provocativeness", index=False)
            if not df_leaderboard.empty:
                df_leaderboard.to_excel(writer, sheet_name="Leaderboard", index=False)

        print(f"Excel‑файл '{metrics_table_filename}' успешно создан (динамически).")
