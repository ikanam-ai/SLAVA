import logging

import pandas as pd
from huggingface_hub import hf_hub_download

from slava.config import OPEN_DATASET_FILENAME, REPO_ID, REPO_TYPE, REQUIRED_COLUMNS


class DataLoader:
    """A class to load datasets from a local path or a remote repository."""

    def __init__(
        self,
        repo_id: str = REPO_ID,
        filename: str = OPEN_DATASET_FILENAME,
    ):
        """
        Initializes the DataLoader.

        Args:
            repo_id (str): The repository ID for loading from Hugging Face Hub.
            filename (str): The filename for loading from Hugging Face Hub.
        """
        self.repo_id = repo_id
        self.filename = filename

    def load_data(self, dataset_path: str = None) -> pd.DataFrame:
        """Loads data from a specified source.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            ValueError: If no parameters are provided for loading data.
            ValueError: If the loaded dataset does not match the required format.
        """
        # Local
        if dataset_path:
            logging.info(f"Dataset loaded from local by path - {dataset_path}")
            data = pd.read_json(dataset_path, lines=True)
        # HuggingFace
        elif self.repo_id and self.filename:
            logging.info(f"Dataset loaded from HuggingFace by path - {self.repo_id}/{self.filename}")
            data = pd.read_json(
                hf_hub_download(repo_id=self.repo_id, filename=self.filename, repo_type=REPO_TYPE),
                lines=True,
            )
        else:
            raise ValueError("Не указаны параметры для загрузки данных.")

        # Validate the dataset format
        if not self.validate_format(data):
            raise ValueError("Загруженный датасет не соответствует требуемому формату.")
        else:
            logging.info("The initial validation has been completed")
        return data

    def validate_format(self, data: pd.DataFrame) -> bool:
        """Checks if the dataset conforms to the required format.

        Args:
            data (pd.DataFrame): The loaded dataset.

        Returns:
            bool: True if the dataset is in the required format, False otherwise.
        """
        return all(column in data.columns for column in REQUIRED_COLUMNS)
