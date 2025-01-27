import tempfile
from typing import Dict, List, Optional, TypedDict, cast

import requests
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from PyPDF2 import PdfReader


class PdfParseInput(TypedDict):
    pdf_link: str


class PdfParseOutput(BaseModel):
    success: bool
    parsed_text: List[str]
    message: Optional[str] = None


def run_pdf_parsing_chain(headers: Optional[Dict[str, str]] = None) -> Runnable[PdfParseInput, PdfParseOutput]:
    """
    Создаёт цепочку для извлечения текста из PDF-документа.

    Args:
        headers (Optional[Dict[str, str]]): Дополнительные HTTP-заголовки (если нужны).

    Returns:
        Runnable[PdfParseInput, PdfParseOutput]: Цепочка для парсинга PDF.
    """

    class PdfParsingRunnable(Runnable[PdfParseInput, PdfParseOutput]):
        def invoke(self, input_data: PdfParseInput) -> PdfParseOutput:
            """
            Выполняет логику извлечения текста из PDF.

            Args:
                input_data (PdfParseInput): Данные с ссылкой на PDF-документ.

            Returns:
                PdfParseOutput: Результат парсинга PDF.
            """
            pdf_link = input_data["pdf_link"]

            try:
                # Извлечение текста из PDF
                reader = PdfReader(pdf_link)
                parsed_text = [page.extract_text() for page in reader.pages]

                # Проверка, что текст извлечён
                if not parsed_text:
                    return PdfParseOutput(
                        success=False,
                        parsed_text=[],
                        message="Не удалось извлечь текст из PDF. Проверьте содержимое документа.",
                    )

                return PdfParseOutput(
                    success=True,
                    parsed_text=parsed_text,
                    message="Текст успешно извлечён из PDF.",
                )
            except requests.exceptions.RequestException as e:
                return PdfParseOutput(
                    success=False,
                    parsed_text=[],
                    message=f"Ошибка загрузки PDF: {str(e)}",
                )
            except Exception as e:
                return PdfParseOutput(
                    success=False,
                    parsed_text=[],
                    message=f"Ошибка обработки PDF: {str(e)}",
                )

    # Оборачиваем в Runnable и приводим к правильному типу
    pdf_parsing_runnable = cast(Runnable[PdfParseInput, PdfParseOutput], PdfParsingRunnable())
    return pdf_parsing_runnable
