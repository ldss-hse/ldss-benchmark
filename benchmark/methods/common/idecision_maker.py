from typing import Optional

from benchmark.task.assessments.decision_matrix import DecisionMatrix
from benchmark.task.task_model import TaskModel


class IDecisionMaker:
    _task: TaskModel
    _decision_matrices: dict[str, DecisionMatrix]
    _criteria_weights: Optional[list[float]]
    _criteria_types: Optional[tuple[bool]]

    def __init__(self, task: TaskModel):
        self._task = task
        self._decision_matrices = self._task.decision_matrices
        self._criteria_weights = self._task.get_criteria_weights()
        self._criteria_types = self._task.get_criteria_types()

    @property
    def decision_matrices(self):
        return self._decision_matrices

    @decision_matrices.setter
    def decision_matrices(self, value):
        self._decision_matrices = value

    def run(self):
        return NotImplemented

    def _normalize_matrices(self):
        for _, matrix in self._decision_matrices.items():
            matrix.normalize()

    def _apply_criteria_weights(self):
        for _, matrix in self._decision_matrices.items():
            matrix.apply_criteria_weights(self._criteria_weights)

    def get_first_decision_matrix(self):
        # WARNING: should be used only for cases when multiple decision matrices are not supported
        return self._decision_matrices[list(self._decision_matrices.keys())[0]]