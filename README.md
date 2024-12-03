# GPTExplainer Module

The `gptexplainer` module provides a `GPTExplainer` class to generate human-readable explanations for model predictions using a GPT model. This module is particularly useful for interpreting predictions by detailing the most influential features and presenting insights in a specified language.

## GPTExplainer

The `GPTExplainer` class creates an instance for interpreting prediction results in a structured, natural-language format.

### Features

1. **Initialization (`__init__`)**:
    - `task_desc`: a description of the prediction model’s task.
    - `preds_descrs`: a dictionary containing predictor names and their descriptions.
    - `gpt_fn`: a function to call the GPT model for generating explanations.
    - `answer_language`: a language for the explanation.

2. **Explaining (`explain`)**:
    - `preds_values`: a dictionary containing predictors and their influence values, where each key is a predictor name and the value is its influence value.