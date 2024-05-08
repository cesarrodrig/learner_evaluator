import pandas as pd
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

from app.constants import DATA_DIR, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from app.learner_evaluator import _format_learning_unit

learning_unit_df = pd.read_csv(DATA_DIR / "content_description.csv")
learning_unit_df = learning_unit_df.assign(learning_unit=learning_unit_df.content_id)

documents = [
    Document(
        page_content=_format_learning_unit(row),
        metadata=row[["learning_unit", "content_title", "subject", "grade", "strand", "substrand"]].to_dict(),
    )
    for _, row in learning_unit_df.iterrows()
]

learning_unit_vectorstore = Neo4jVector(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    password=NEO4J_PASSWORD,
    username=NEO4J_USERNAME,
)
