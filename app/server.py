from operator import itemgetter
from typing import TypedDict

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes

from app.constants import DATA_DIR, MODELS_DIR, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from app.knowledge_graph import build_cypher_qa_chain
from app.learner_evaluator import build_learner_evaluator
from app.repository import ActivityPandasRepository, LearnerPandasRepository, LearningUnitPandasRepository

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    # model="gpt-4-0125-preview",
)
set_llm_cache(SQLiteCache())


learning_unit_vectorstore = Neo4jVector(
    index_name="learning_units",
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    password=NEO4J_PASSWORD,
    username=NEO4J_USERNAME,
)
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

activity_df = pd.read_parquet(DATA_DIR / "processed_learning_activity.parquet")
learning_unit_df = pd.read_csv(DATA_DIR / "content_description.csv")
learner_df = pd.read_csv(DATA_DIR / "learner_details.csv")

activity_repo = ActivityPandasRepository(activity_df)
learning_unit_repo = LearningUnitPandasRepository(learning_unit_df)
learner_repo = LearnerPandasRepository(learner_df)

regressor = xgb.XGBRegressor(enable_categorical=True, validate_parameters=True)
regressor.load_model(MODELS_DIR / "xgboost_regressor.json")

learner_evaluator = build_learner_evaluator(
    llm,
    learning_unit_vectorstore,
    activity_repo,
    learning_unit_repo,
    learner_repo,
    regressor,
)
query_knowledge_graph = build_cypher_qa_chain(graph=graph, llm=llm)


class EvaluateLearnerInput(TypedDict):
    learner_id: str


class EvaluateLearnerOutput(TypedDict):
    output: str


class QueryKnowledgeGraphInput(TypedDict):
    query: str


class QueryKnowledgeGraphOutput(TypedDict):
    answer: str


add_routes(
    app,
    learner_evaluator.with_types(input_type=EvaluateLearnerInput, output_type=EvaluateLearnerOutput),
    path="/evaluate_learner",
)


add_routes(
    app,
    (query_knowledge_graph | itemgetter("result")).with_types(
        input_type=QueryKnowledgeGraphInput, output_type=QueryKnowledgeGraphOutput
    ),
    path="/query_knowledge_graph",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
