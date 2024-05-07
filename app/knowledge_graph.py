import json
from pathlib import Path

from langchain.chains import GraphCypherQAChain
from langchain.chains.base import Chain
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from app.constants import DATA_DIR, PROMPTS_DIR


def build_cypher_qa_chain(graph: Neo4jGraph, llm: LLM) -> Chain:

    examples = json.loads(Path(DATA_DIR / "cypher_query_examples.json").read_text())
    example_prompt = PromptTemplate.from_template("User input: {question}\nCypher query: {query}")
    build_cypher_query_prompt = Path(PROMPTS_DIR / "build_cypher_query").read_text()

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=build_cypher_query_prompt,
        suffix="User input: {question}\nCypher query: ",
        input_variables=["question", "schema"],
    )
    return GraphCypherQAChain.from_llm(graph=graph, llm=llm, cypher_prompt=prompt)
