<div align="center">
  <a href="https://huggingface.co/spaces/RANEPA-ai/SLAVA"><img src="extensions/views/logo.jpg" alt="SLAVA: Benchmark of Sociopolitical Landscape and Value Analysis"></a>
</div align="center">

# SLAVA: Benchmark of Sociopolitical Landscape and Value Analysis

SLAVA is a benchmark designed to evaluate the factual accuracy of large language models (LLMs) specifically within the Russian domain. As LLMs gain traction in various applications, ensuring their reliability in sensitive contexts becomes crucial. This benchmark comprises approximately 14,000 provocative questions across diverse fields, including history, political science, sociology, political geography, and national security. Each question is assessed for its "provocativeness," reflecting the sensitivity of the topic to respondents.

## Testing models on our open data (DRAFT)

Here are instructions for generating predictions of a model from the SLAVE leaderboard on our open data.

1. Get an open dataset from our hugging face page

2. Generate your model's responses in the "response" column

3. Save the data to a table and upload it to our model validation module

## Code structure of the framework
```
├── LICENSE            <- Open-source license
├── README.md          <- README for developers using this framework.
│
├── extensions
│   └── views          <- Images and graphic objects
│
├── .gitignore         <- The .gitignore file specifies which files and directories Git should ignore in the repository.
│
├── poetry.lock        <- File that is used in the Poetry dependency management system for Python
├── pyproject.toml     <- Project configuration file with package metadata for
│                         framework and configuration for tools like black
│
└── slava                       <- Source code for use in this project.
    ├── __init__.py
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── modules
    │   ├── __init__.py
    │   └── metrics.py          <- Сode for getting metrics for the selected experiment
```
