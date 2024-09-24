import pandas as pd
from huggingface_hub import hf_hub_download

from slava.config import OPEN_DATASET_FILENAME, REPO_ID, REPO_TYPE


class DataLoader:
    """A class to load datasets from a local path or a remote repository."""

    def __init__(
        self,
        dataset_path: str = None,
        repo_id: str = REPO_ID,
        filename: str = OPEN_DATASET_FILENAME,
    ):
        """
        Initializes the DataLoader.

        Args:
            dataset_path (str): The local path to the dataset file.
            repo_id (str): The repository ID for loading from Hugging Face Hub.
            filename (str): The filename for loading from Hugging Face Hub.
        """
        self.dataset_path = dataset_path
        self.repo_id = repo_id
        self.filename = filename

    def load_data(self) -> pd.DataFrame:
        """Loads data from a specified source.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            ValueError: If no parameters are provided for loading data.
        """
        if self.dataset_path:
            return pd.read_json(self.dataset_path)

        elif self.repo_id and self.filename:
            return pd.read_json(
                hf_hub_download(
                    repo_id=self.repo_id, filename=self.filename, repo_type=REPO_TYPE
                )
            )
        else:
            raise ValueError("Не указаны параметры для загрузки данных.")


# Usage example (uncomment below to use)
# if __name__ == "__main__":
#     data_loader = DataLoader()
#     data = data_loader.load_data()
#     print(data.head())
