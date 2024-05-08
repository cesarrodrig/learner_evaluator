from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Union

import xgboost as xgb
from langchain.chains.base import Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, chain
from langchain_core.vectorstores import VectorStore

from app.constants import PROMPTS_DIR
from app.errors import LearnerIDNotFound, NoLearnerActivityError
from app.repository import ActivityPandasRepository, LearnerPandasRepository, LearningUnitPandasRepository


def build_learner_evaluator(
    llm: LLM,
    learning_unit_vectorstore: VectorStore,
    activity_repo: ActivityPandasRepository,
    learning_unit_repo: LearningUnitPandasRepository,
    learner_repo: LearnerPandasRepository,
    score_regressor: xgb.XGBRegressor,
) -> Chain:
    evaluate_learner_on_unit = _build_evaluate_learner_on_unit_chain(
        llm=llm,
        score_regressor=score_regressor,
        activity_repo=activity_repo,
        learning_unit_repo=learning_unit_repo,
    )

    evaluate_learner_on_every_unit = _fanout_chain_on_learning_units(evaluate_learner_on_unit, learning_unit_repo)

    summarize_all_learning_units_results_prompt = _load_prompt_template("summarize_all_learning_units_results")
    summarize_all_learning_units_results = create_stuff_documents_chain(
        llm, summarize_all_learning_units_results_prompt, document_separator="\n=================\n"
    )

    learning_unit_retriever = learning_unit_vectorstore.as_retriever()

    suggest_next_learning_unit_prompt = _load_prompt_template("suggest_next_learning_unit")
    suggest_next_learning_unit = suggest_next_learning_unit_prompt | llm | StrOutputParser()
    no_learner_found = RunnableLambda(lambda inputs: f"Learner with ID {inputs['learner_id']} does not exist.")

    return (
        evaluate_learner_on_every_unit
        | {
            "context": itemgetter("documents") | RunnablePassthrough(),
            "learner_details": itemgetter("learner_id") | _get_learner_details_chain(learner_repo),
        }
        | {
            "knowledge_state": summarize_all_learning_units_results,
            "learner_details": itemgetter("learner_details") | RunnablePassthrough(),
        }
        | {
            "relevant_learning_units": itemgetter("knowledge_state") | learning_unit_retriever,
            "learner_details": itemgetter("learner_details") | RunnablePassthrough(),
            "knowledge_state": itemgetter("knowledge_state") | RunnablePassthrough(),
        }
        | {
            "next_learning_unit": suggest_next_learning_unit,
            "learner_details": itemgetter("learner_details") | RunnablePassthrough(),
            "knowledge_state": itemgetter("knowledge_state") | RunnablePassthrough(),
        }
        | _load_prompt_template("final_output")
        | attrgetter("text")
    ).with_fallbacks([no_learner_found], exceptions_to_handle=[LearnerIDNotFound])


def _build_evaluate_learner_on_unit_chain(
    llm: LLM,
    score_regressor: xgb.XGBRegressor,
    activity_repo: ActivityPandasRepository,
    learning_unit_repo: LearningUnitPandasRepository,
) -> Chain:
    regressor_runnable = _build_regressor_runnable(score_regressor, activity_repo)

    learning_unit_mean_score = RunnableLambda(
        lambda inputs: float(activity_repo.get(inputs["learning_unit_id"]).score.mean())
    )
    learning_unit_std_score = RunnableLambda(
        lambda inputs: float(activity_repo.get(inputs["learning_unit_id"]).score.std())
    )
    no_activity_found = RunnableLambda(lambda _: "No past activity found for learner.")
    learner_mean_score = RunnableLambda(
        lambda inputs: float(activity_repo.get(inputs["learning_unit_id"], inputs["learner_id"]).score.mean())
    ).with_fallbacks([no_activity_found], exceptions_to_handle=[NoLearnerActivityError])

    learner_scores_and_unit_descriptions = RunnableParallel(
        {
            "learning_unit_description": _get_learning_unit_description_chain(learning_unit_repo),
            "learner_pred_score": regressor_runnable | _round_if_number,
            "learner_mean_score": learner_mean_score | _round_if_number,
            "learning_unit_mean_score": learning_unit_mean_score | _round_if_number,
            "learning_unit_std_score": learning_unit_std_score | _round_if_number,
        }
    )

    summarize_learning_unit_prompt = _load_prompt_template("summarize_learning_unit_results")
    return learner_scores_and_unit_descriptions | summarize_learning_unit_prompt | llm | StrOutputParser()


def _build_regressor_runnable(regressor: xgb.XGBRegressor, activity_repo: ActivityPandasRepository) -> RunnableLambda:

    def _predict(inputs: dict) -> float:
        """Predict the score for a learner on a learning unit. If no activity is found, return the average score."""
        learner_id = inputs["learner_id"]
        learning_unit = inputs["learning_unit_id"]
        activities = activity_repo.get(learning_unit, learner_id)

        last_activity = activities.tail(1)
        features = last_activity[regressor.get_booster().feature_names]
        return float(regressor.predict(features)[0])

    def _fallback(inputs: dict) -> str:
        learner_id = inputs["learner_id"]
        activities = activity_repo.get(learner_id=learner_id)

        last_activity = activities.tail(1)
        features = last_activity[regressor.get_booster().feature_names]
        predicted_score = regressor.predict(features)[0]
        return f"No past activity found in this unit. Predicted score based on last activity: {predicted_score:.02f}"

    return RunnableLambda(_predict).with_types(output_type=float).with_fallbacks([RunnableLambda(_fallback)])


def _fanout_chain_on_learning_units(chain: Chain, learning_unit_repo: LearningUnitPandasRepository) -> RunnableLambda:

    def _call_on_learning_units(inputs: dict):
        learner_id = inputs["learner_id"]
        documents = []
        for learning_unit in learning_unit_repo.list():
            learning_unit_id = learning_unit["learning_unit_id"]
            learning_unit_description = _format_learning_unit(learning_unit_repo.get(learning_unit_id))
            results = chain.invoke({"learner_id": learner_id, "learning_unit_id": learning_unit_id})
            page_content = f"""{learning_unit_description}\n\nLearner Knowledge State Summary:\n{results}"""
            documents.append(Document(page_content=page_content))
        return {"learner_id": learner_id, "documents": documents}

    return RunnableLambda(_call_on_learning_units)


def _get_learning_unit_description_chain(learning_unit_repo: LearningUnitPandasRepository) -> RunnableLambda:

    @chain
    def _get_and_format_learning_unit(inputs: dict) -> dict:
        learning_unit = learning_unit_repo.get(inputs["learning_unit_id"])
        return _format_learning_unit(learning_unit)

    return _get_and_format_learning_unit


def _get_learner_details_chain(learner_repo: LearnerPandasRepository) -> RunnableLambda:
    learner_template = Path(PROMPTS_DIR / "learner_details").read_text()

    @chain
    def _format_learner(learner_id: str) -> dict:
        learner = learner_repo.get(learner_id)
        return learner_template.format(**learner)

    return _format_learner


def _load_prompt_template(prompt_name: str, prompts_dir: Path = PROMPTS_DIR) -> PromptTemplate:
    template_path = prompts_dir / prompt_name
    return PromptTemplate.from_template(template_path.read_text())


def _format_learning_unit(learning_unit: dict) -> str:
    learning_unit_template = Path(PROMPTS_DIR / "learning_unit").read_text()
    return learning_unit_template.format(**learning_unit)


@chain
def _round_if_number(number: Union[float, str]) -> Union[float, str]:
    if isinstance(number, float):
        return round(number, 2)
    return number
