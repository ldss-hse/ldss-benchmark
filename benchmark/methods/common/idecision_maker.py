from typing import Optional

import numpy as np

from benchmark.task.assessments.decision_matrix import DecisionMatrix
from benchmark.task.task_model import TaskModel


class IDecisionMaker:
    _task: TaskModel
    _decision_matrices: dict[str, DecisionMatrix]
    _criteria_weights: Optional[list[float]]
    _criteria_types: Optional[tuple[bool]]
    _joint_decision_matrix: Optional[DecisionMatrix]

    def __init__(self, task: TaskModel):
        self._task = task
        self._decision_matrices = self._task.decision_matrices
        self._joint_decision_matrix = None
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

    def _normalize_matrix(self):
        self._joint_decision_matrix.normalize()

    def _apply_criteria_weights(self):
        self._joint_decision_matrix.apply_criteria_weights(self._criteria_weights)

    def _aggregate_assessments_by_experts(self):
        if self._joint_decision_matrix is not None:
            return
        acc = np.zeros_like(self._decision_matrices[self._task._dto.experts[0].expertID].get_raw())

        weights = self._task._dto.expertWeights
        if weights is None:
            weights = self._decide_expert_weights()

        for expert_name, expert_assessments in self._decision_matrices.items():
            acc += expert_assessments.get_raw() * weights[expert_name]
        self._joint_decision_matrix = DecisionMatrix(raw_assessments=np.round(acc, 4))

    def _decide_expert_weights(self):
        return NotImplemented
