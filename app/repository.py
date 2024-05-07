from typing import Optional

import pandas as pd


class AbstractRepository:

    def get(self, *args, **kwargs):
        raise NotImplementedError

    def list(self, *args, **kwargs):
        raise NotImplementedError


class ActivityPandasRepository(AbstractRepository):

    def __init__(self, activity_df: pd.DataFrame):
        activity_df = activity_df.sort_values("play_time")
        self.activity_df = activity_df

    def get(self, learning_unit: Optional[str] = None, learner_id: Optional[str] = None) -> pd.DataFrame:
        learner_activity_df = self.activity_df
        if learning_unit:
            learner_activity_df = learner_activity_df[learner_activity_df.learning_unit == learning_unit]
        if learner_id:
            learner_activity_df = learner_activity_df[learner_activity_df.learner_id == learner_id]
        if learner_activity_df.empty:
            raise RuntimeError(f"No past activity found for learner {learner_id} on learning unit {learning_unit}.")

        return learner_activity_df


class LearningUnitPandasRepository(AbstractRepository):

    def __init__(self, learning_unit_df: pd.DataFrame):
        self.learning_unit_df = learning_unit_df.assign(learning_unit_id=learning_unit_df.content_id)

    def get(self, learning_unit_id: str) -> dict:
        if learning_unit_id not in self.learning_unit_df.content_id.values:
            print(f"Learning unit {learning_unit_id} not found.")
        return self.learning_unit_df[self.learning_unit_df.learning_unit_id == learning_unit_id].iloc[0].to_dict()

    def list(self) -> list[dict]:
        return self.learning_unit_df.to_dict(orient="records")

    def get_substrand(self, learning_unit: str) -> str:
        return self.learning_unit_df[self.learning_unit_df.content_id == learning_unit].substrand.values[0]


class LearnerPandasRepository(AbstractRepository):

    def __init__(self, learner_df: pd.DataFrame):
        self.learner_df = learner_df

    def get(self, learner_id: str) -> dict:
        return self.learner_df[self.learner_df.learner_id == learner_id].iloc[0].to_dict()
