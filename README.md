# Learner Evaluator

Modeling learner activity to predict scores on learning units.

## Setup

Install dependencies:
```bash
poetry install
```

Populate the storages:
```bash
# In separate terminal
docker-compose up
```

Then:
```bash
poetry run python scripts/init_knowledge_graph.py
poetry run python scripts/init_vector_store.py
```

Export OpenAI API key available
```bash
export OPENAI_API_KEY="$(cat OpenAIKey.txt)"
```

## Usage

See `modeling/static_modeling.ipynb` for the experiments of a few models.

Run the Learner Evaluator:
```bash
poetry run langchain serve --port=8100
```

Go to http://localhost:8100/evaluate_learner/playground to run queries on learners.

Go to http://localhost:8100/query_knowledge_graph/playground to run queries on the knowledge graph.
