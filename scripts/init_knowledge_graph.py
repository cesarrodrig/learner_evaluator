# %%%
# !pip install --upgrade --quiet  langchain langchain-community langchain-openai langchain-experimental neo4j
# %%%

import os
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

from app.constants import DATA_DIR

os.environ["OPENAI_API_KEY"] = Path("OpenAIKey.txt").read_text().strip()

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

activity_df = pd.read_parquet(DATA_DIR / "processed_learning_activity.parquet")
learning_unit_df = pd.read_csv(DATA_DIR / "content_description.csv")


driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
driver.verify_connectivity()


def _create_nodes(df, node_name: str, id_col: str):
    # # unique_constraint_query = f"""
    # # CREATE CONSTRAINT {node_name}_id_unique
    # # FOR (n:`{node_name}`)
    # # REQUIRE n.id IS UNIQUE
    # # """
    # try:
    #     driver.execute_query(unique_constraint_query, database_="neo4j")
    # except RuntimeError:
    #     pass

    node_query = (
        f"""
    WITH $node_ids AS batch
    UNWIND batch AS node
    MERGE (n:`{node_name}`"""
        """{id: node.id})"""
    )
    node_ids = [{"id": node_id} for node_id in df[id_col].unique()]
    print(f"Creating {len(node_ids)} nodes")
    driver.execute_query(node_query, node_name=node_name, node_ids=node_ids, database_="neo4j")


def _create_relation(df, relation_name: str, from_id_col: str, to_id_col: str, from_node: str, to_node: str):
    relation_query = f"""
    MATCH (from:`{from_node}`), (to:`{to_node}`)
    WHERE from.id = $from_id_col AND to.id = $to_id_col
    CREATE (from)-[:{relation_name}]->(to)
    """

    df = df[[from_id_col, to_id_col]].drop_duplicates()
    for _, row in df.iterrows():
        driver.execute_query(
            relation_query,
            relation_name=relation_name,
            from_id_col=row[from_id_col],
            to_id_col=row[to_id_col],
            from_node=from_node,
            to_node=to_node,
            database_="neo4j",
        )


_create_nodes(activity_df, node_name="Learner", id_col="learner_id")
_create_nodes(activity_df, node_name="School", id_col="school_id")
_create_nodes(activity_df, node_name="Class", id_col="class_id")
_create_nodes(activity_df, node_name="Learning Unit", id_col="learning_unit")
_create_nodes(activity_df, node_name="Activity", id_col="id")
_create_nodes(activity_df, node_name="Grade", id_col="grade")

# create the following relations: Student has Activity Summaries
# Learning Unit has Exercises
# Learning Unit improves Target Skills
# School has Classes
# Class has Students
# Activity Summary was done by Student
# Activity Summary pertains to Learning Unit

_create_relation(
    activity_df,
    relation_name="ATTENDS_SCHOOL",
    from_id_col="learner_id",
    to_id_col="school_id",
    from_node="Learner",
    to_node="School",
)

_create_relation(
    activity_df,
    relation_name="FROM_LEARNING_UNIT",
    from_id_col="id",
    to_id_col="learning_unit",
    from_node="Activity",
    to_node="Learning Unit",
)

_create_relation(
    activity_df,
    relation_name="HAS_CLASS",
    from_id_col="school_id",
    to_id_col="class_id",
    from_node="School",
    to_node="Class",
)

_create_relation(
    activity_df,
    relation_name="ATTENDS_CLASS",
    from_id_col="learner_id",
    to_id_col="class_id",
    from_node="Learner",
    to_node="Class",
)

_create_relation(
    activity_df,
    relation_name="PERFORMED_ACTIVITY",
    from_id_col="learner_id",
    to_id_col="id",
    from_node="Learner",
    to_node="Activity",
)

_create_relation(
    activity_df,
    relation_name="PERFORMED_BY_LEARNER",
    from_id_col="id",
    to_id_col="learner_id",
    from_node="Activity",
    to_node="Learner",
)

_create_relation(
    activity_df,
    relation_name="OF_GRADE",
    from_id_col="learning_unit",
    to_id_col="id",
    from_node="Learning Unit",
    to_node="Grade",
)
_create_relation(
    activity_df,
    relation_name="IN_GRADE",
    from_id_col="learner_id",
    to_id_col="id",
    from_node="Learner",
    to_node="Grade",
)


def _add_properties_to_node(df, node_name: str, id_col: str, properties: list):
    add_query = (
        f"""
    WITH $nodes AS batch
    UNWIND batch AS node
    WITH node, properties(node) as props
    MERGE (n:`{node_name}` """
        """{id: props.%s}) SET n += props RETURN n""" % id_col
    )
    print(add_query)
    print([id_col] + properties)
    node_as_dicts = df[[id_col] + properties].drop_duplicates().to_dict(orient="records")
    driver.execute_query(
        add_query,
        nodes=node_as_dicts,
        database_="neo4j",
    )


_add_properties_to_node(
    activity_df,
    node_name="Learner",
    id_col="learner_id",
    properties=["gender"],
)

_add_properties_to_node(
    activity_df,
    node_name="Activity",
    id_col="id",
    properties=["play_time", "outcome", "foreground_duration", "score"],
)

_add_properties_to_node(
    learning_unit_df.rename(columns={"content_id": "id"}),
    node_name="Learning Unit",
    id_col="id",
    properties=["subject", "grade", "strand", "substrand", "content_title", "exercise_type", "content_description"],
)


def _add_agg_features_from_df(activity_df: pd.DataFrame, from_df: pd.DataFrame, row_index: int, col_prefix: str):
    for agg_function in ["mean", "std", "min", "max"]:
        activity_df.loc[row_index, f"{col_prefix}_scores_{agg_function}"] = from_df.score.agg(agg_function)
        activity_df.loc[row_index, f"{col_prefix}_foreground_durations_{agg_function}"] = (
            from_df.foreground_duration.agg(agg_function)
        )

    for outcome in activity_df.outcome.unique():
        is_outcome = from_df.outcome == outcome

        outcome_count = is_outcome.sum()
        col_name = f"{col_prefix}_{outcome.lower()}_count"
        activity_df.loc[row_index, col_name] = outcome_count

        past_outcome_rate = outcome_count / (len(is_outcome) + 1e-6)
        col_name = f"{col_prefix}_{outcome.lower()}_rate"
        activity_df.loc[row_index, col_name] = past_outcome_rate


for learning_unit in learning_unit_df.content_id.unique():
    learning_unit_activity_df = activity_df[activity_df.learning_unit == learning_unit]
    _add_agg_features_from_df(
        activity_df, learning_unit_activity_df, learning_unit_activity_df.index, col_prefix="learning_unit"
    )

properties = [col_name for col_name in activity_df.columns if "learning_unit_" in col_name]
_add_properties_to_node(
    activity_df.drop(columns=["id"]).rename(columns={"learning_unit": "id"}),
    node_name="Learning Unit",
    id_col="id",
    properties=properties,
)

for learner_id in activity_df.learner_id.unique():
    learner_activity_df = activity_df[activity_df.learner_id == learner_id]
    _add_agg_features_from_df(activity_df, learner_activity_df, learner_activity_df.index, col_prefix="learner")

properties = [col_name for col_name in activity_df.columns if "learner_" in col_name and col_name != "learner_id"]
_add_properties_to_node(
    activity_df.drop(columns=["id"]).rename(columns={"learner_id": "id"}),
    node_name="Learner",
    id_col="id",
    properties=properties,
)
