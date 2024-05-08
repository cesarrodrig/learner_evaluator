import pandas as pd
from langchain import smith
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langsmith import Client

from app import config
from app.constants import DATA_DIR
from app.learner_evaluator import _load_prompt_template


def main():
    llm = ChatOpenAI(temperature=config.LLM_TEMPERATURE, model=config.LLM)

    summarize_learning_unit_prompt = _load_prompt_template("summarize_learning_unit_results")
    summarize_learning_unit = summarize_learning_unit_prompt | llm | StrOutputParser()

    df = pd.read_json(DATA_DIR / "tests" / "summarize_learning_unit_results.json", orient="records")
    columns = df.columns
    input_keys = columns.drop("output").tolist()
    output_keys = ["output"]

    client = Client()
    dataset_name = "summarize_learning_unit_results"
    if not client.has_dataset(dataset_name=dataset_name):
        client.upload_dataframe(
            df=df,
            input_keys=input_keys,
            output_keys=output_keys,
            name=dataset_name,
            data_type="kv",
        )

    eval_config = smith.RunEvalConfig(
        evaluators=[
            smith.RunEvalConfig.LabeledCriteria("conciseness", input_key="id", output_key="output"),
            smith.RunEvalConfig.EmbeddingDistance(input_key="id", output_key="output"),
            smith.RunEvalConfig.LabeledScoreString(
                {
                    "accuracy": """
Score 1: The assessment is statiscally incorrect.
Score 3: The assessment has statistical flaws but it does not align with the answer.
Score 7: The assessment is almost statistically correct and it  align with the answer.
Score 10: The assessment is statiscally correct and makes sense with the numbers provided"""
                },
                output_key="output",
                input_key="id",
                normalize_by=10,
            ),
        ],
        eval_llm=ChatOpenAI(model="gpt-4", temperature=0),
    )

    chain_results = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=lambda inputs: summarize_learning_unit.invoke(inputs),
        evaluation=eval_config,
        concurrency_level=5,
        verbose=True,
    )

    print("Experiment Name:", chain_results["project_name"])


if __name__ == "__main__":
    main()
