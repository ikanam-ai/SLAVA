import logging

import pandas as pd
from huggingface_hub import hf_hub_download

from slava.config import OPEN_DATASET_FILENAME, REPO_ID, REPO_TYPE, REQUIRED_COLUMNS


class DataLoader:

    def __init__(
        self,
        repo_id: str = REPO_ID,
        filename: str = OPEN_DATASET_FILENAME,
    ):
        self.repo_id = repo_id
        self.filename = filename

    def load_data(self, dataset_path: str = None) -> pd.DataFrame:
        # Local JSONL
        if dataset_path:
            logging.info(f"Dataset loaded from local by path - {dataset_path}")
            data = pd.read_json(dataset_path, lines=True)
        # HuggingFace JSONL
        elif self.repo_id and self.filename:
            logging.info(f"Dataset loaded from HuggingFace by path - {self.repo_id}/{self.filename}")
            data = pd.read_json(
                hf_hub_download(repo_id=self.repo_id, filename=self.filename, repo_type=REPO_TYPE),
                lines=True,
            )
        else:
            raise ValueError("The parameters for loading data are not specified")

        # Validate the dataset format
        if not self.validate_format(data):
            raise ValueError("The uploaded dataset does not match the required format")
        else:
            logging.info("The initial validation has been completed")
        return data

    def validate_format(self, data: pd.DataFrame) -> bool:
        return all(column in data.columns for column in REQUIRED_COLUMNS)
