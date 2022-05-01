from pathlib import Path

import numpy as np

from .assessments.decision_matrix import DecisionMatrix
from .schemas.task_scheme import TaskDTOScheme


class TaskModel:
    _dto: TaskDTOScheme
    _decision_matrix: DecisionMatrix

    def __init__(self, dto):
        self._dto = dto
        self._decision_matrix = self.extract_raw_decision_matrix()

    def extract_raw_decision_matrix(self):
        raw_assessments = []
        for expert_id, expert_assessments in self._dto.estimations.items():
            print(expert_id)

        return DecisionMatrix(np.array(raw_assessments))



class TaskModelFactory:
    def from_json(self, json_path: Path):

        with json_path.open(encoding='utf-8') as task_file:
            task_raw = task_file.read()

        task_parsed = TaskDTOScheme.__pydantic_model__.parse_raw(task_raw)
        task_model = TaskModel(task_parsed)
        print(task_parsed)
        return task_model
