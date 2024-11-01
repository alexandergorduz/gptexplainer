from typing import Dict



class GPTExplainer:
    """
    Class GPTExplainer for creating GPT explainer instance.
    """


    def __init__(self, task_desc: str, preds_descrs: Dict,
                 gpt_fn, answer_language: str = "English") -> None:
        """
        Initialize the GPTExplainer class.
        
        Args:
            task_desc (str): description of the task the prediction model is working on.
            preds_descrs (Dict): a dictionary with predictors and their descriptions.
            gpt_fn (function): a function that calls the GPT model.
            answer_language (str, optional): the language in which we want to receive the explanation.
        """

        self.task_desc = task_desc
        self.preds_descrs = preds_descrs
        self.gpt_fn = gpt_fn
        self.answer_language = answer_language

        self.prompt_template = """You are a Data Scientist who understands how to interpret model results.

The task of the model:
'{task_desc}'

You have a list of the features names with influences values of specific prediction and features descriptions:
{prediction_influences}
The larger absolute influence value the more this feature influenced the specific prediction.
Carefully consider how features influenced the specific prediction.
Draw a conclusion in one or two sentences.

**Content Restrictions:**
1. **Provide the response in {answer_language} language.**
2. **Omit features if they have little influence.**
3. **Do not mention influence values.**
4. **Do not respond with numbered list.**
5. **Do not start your response with regular phrases.**
6. **Do not include your own additional conclusions about model or task.**
7. **Important to include the ORIGINAL features names in parentheses when mentioning them.**"""


    def explain(self, preds_values: Dict) -> str:
        """
        Performs an explaining specific prediction using influence values.
        
        Args:
            preds_values (Dict): a dictionary with predictors and their influences values,
                where the key is the predictor name and the value is the predictor influence value.
        
        Returns:
            str: the text of explaining specific prediction.
        """

        if not all(pred in self.preds_descrs for pred in preds_values):
            raise Exception(
                "Some incomed predictors unacceptable."
            )
        
        prediction_influences = ""

        for pred, descr in self.preds_descrs.items():

            prediction_influences += f"- {pred} = {round(preds_values[pred], 5)} - {descr}\n"
        
        input_text = self.prompt_template.format(task_desc=self.task_desc,
                                                 prediction_influences=prediction_influences,
                                                 answer_language=self.answer_language)
        
        return self.gpt_fn(input_text).strip().replace("\n", " ")