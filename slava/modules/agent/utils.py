import json
from typing import Annotated, Dict, List, Optional

from huggingface_hub import HfApi

from slava.config import jsonl_filename, repo_id, token

api = HfApi()


def get_jsonl_files(repo_id: str = repo_id, token: str = token) -> List[str]:
    """
    Получение списка файлов с расширением .jsonl из репозитория Hugging Face.

    Args:
        repo_id (str): ID репозитория (по умолчанию из конфигурации).
        token (str): Токен для аутентификации (по умолчанию из конфигурации).

    Returns:
        List[str]: Список файлов .jsonl в репозитории.
    """
    files = api.list_repo_files(repo_id=repo_id, token=token, repo_type="dataset")
    jsonl_files = [file.split(".")[0] for file in files if file.endswith(".jsonl")]
    return jsonl_files


def save_results_to_jsonl(results: List[dict], filename: str = jsonl_filename) -> None:
    """
    Сохраняет результаты в формате JSONL.

    Args:
        results (List[dict]): Список результатов.
        filename (str): Имя файла для сохранения.
    """
    with open(filename, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def upload_to_huggingface(
    topic: str, filename: str = jsonl_filename, repo_id: str = repo_id, token: str = token
) -> str:
    """
    Загружает файл на Hugging Face и возвращает ссылку на репозиторий.

    Args:
        filename (str): Имя файла для загрузки.
        repo_id (str): ID репозитория на Hugging Face (например, "username/dataset_name").
        token (str): Токен аутентификации Hugging Face.
        topic (str): Название темы, под которой будет сохранён файл в репозитории.

    Returns:
        str: Ссылка на загруженный репозиторий.
    """
    api.upload_file(
        path_or_fileobj=filename, path_in_repo=f"{topic}.jsonl", repo_id=repo_id, repo_type="dataset", token=token
    )
    return f"https://huggingface.co/datasets/{repo_id}"
