from typing import Optional

from langchain.evaluation import load_evaluator
from langsmith import evaluation as ls_evaluation
from langsmith.schemas import Example, Run


class ConcisenessEvaluator(ls_evaluation.RunEvaluator):

    def __init__(self):
        self.evaluator = load_evaluator("criteria", criteria="conciseness")

    def evaluate_run(self, run: Run, example: Optional[Example] = None) -> ls_evaluation.EvaluationResult:
        if (
            not run.outputs
            or not run.outputs.get("output")
            or not run.child_runs
            or not run.child_runs[-1].outputs
            or not run.child_runs[-1].outputs.get("output")
        ):
            return ls_evaluation.EvaluationResult(key="conciseness", score=None)

        last_step_output = run.child_runs[-1]
        question = last_step_output.outputs["output"]
        prediction = run.outputs["output"]
        result = self.evaluator.evaluate_strings(input=question, prediction=prediction)
        return ls_evaluation.EvaluationResult(key="conciseness", **result)


class LearningUnitEvaluator:
    def __init__(self, learning_units: list[str]):
        self.learning_units = learning_units

    def evaluate_run(self, run: Run, example: Optional[Example] | None = None) -> ls_evaluation.EvaluationResult:
        model_output = run.outputs["output"]
        score = any([learning_unit in model_output for learning_unit in self.learning_units])
        return ls_evaluation.EvaluationResult(key="has_learning_unit", score=score)
