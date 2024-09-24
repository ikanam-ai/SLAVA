import pandas as pd
from tqdm import tqdm


class ModelEval:
    """A class to evaluate models using a data loader and a model handler."""

    def __init__(self, model_handler, data_loader):
        """
        Initializes the ModelEval.

        Args:
            model_handler: An instance of a model handler for generating responses.
            data_loader: An instance of a data loader for loading data.
        """
        self.model_handler = model_handler
        self.data_loader = data_loader

    def fill_instruction(self, row: pd.Series) -> str:
        """Fills the instruction template with values from the given row.

        Args:
            row (pd.Series): A row from the test data.

        Returns:
            str: The filled instruction string.
        """
        instruction_template = row["instruction"]  # Ensure the column name is correct

        # Create a dictionary of values for substitution
        values = {
            "task": row["inputs"].get("task", ""),  # Use get for safe access
            "text": row["inputs"].get("text", ""),
            "Option_1": row["inputs"]["options"].get("option_1", ""),
            "Option_2": row["inputs"]["options"].get("option_2", ""),
            "Option_3": row["inputs"]["options"].get("option_3", ""),
            "Option_4": row["inputs"]["options"].get("option_4", ""),
            "Option_5": row["inputs"]["options"].get("option_5", ""),
            "Option_6": row["inputs"]["options"].get("option_6", ""),
            "Option_7": row["inputs"]["options"].get("option_7", ""),
            "Option_8": row["inputs"]["options"].get("option_8", ""),
            "Option_9": row["inputs"]["options"].get("option_9", ""),
        }

        # Remove NaN or empty values from the dictionary
        values = {k: v for k, v in values.items() if pd.notna(v) and v != ""}

        # Fill the template
        try:
            filled_instruction = (
                instruction_template.format(**values)
                + "\nСАМОЕ ВАЖНОЕ: Отвечай максимально кратко используя только цифры если они даны или слова в задачах с открытым ответом.\nОтвет: "
            )
        except KeyError as e:
            print(f"Ошибка подстановки: отсутствует ключ {e}")
            filled_instruction = (
                "Ошибка: отсутствует необходимая информация для формирования запроса."
            )

        return filled_instruction

    def run_evaluation(self, output_file: str) -> None:
        """Runs the evaluation and saves results to a file.

        Args:
            output_file (str): The path to the output CSV file where results will be saved.
        """
        # Load test data
        test_data = self.data_loader.load_data()

        # Create a list to save results
        results = []

        # Generate predictions for each question
        for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
            # Fill the prompt using the function
            prompt = self.fill_instruction(row)
            response = self.model_handler.generate_response(prompt)
            results.append(
                {
                    "input": prompt,
                    "response": response.strip(),
                    "output": row["outputs"],
                }
            )

        # Save results in a DataFrame
        results_df = pd.DataFrame(results)

        # Save results to a file
        results_df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Results saved to {output_file}")


# Usage example (uncomment below to use)
# if __name__ == "__main__":
#     model_handler = ...  # Your model handler instance
#     data_loader = ...  # Your data loader instance
#     evaluator = ModelEval(model_handler, data_loader)
#     evaluator.run_evaluation("output.csv")
