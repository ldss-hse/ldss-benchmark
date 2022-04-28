from typing import Optional

from benchmark.assessments.decision_matrix import DecisionMatrix


class IDecisionMaker:
    _decision_matrix: DecisionMatrix
    criteria_weights: Optional[list[float]]
    is_benefit_criteria: Optional[tuple[bool]]

    def __init__(self, decision_matrix: DecisionMatrix):
        self._decision_matrix = decision_matrix
        self.criteria_weights = None

    def set_criteria_weights(self, criteria_weights):
        self.criteria_weights = criteria_weights

    def set_alternatives_type(self, is_benefit_criteria: tuple[bool]):
        self.is_benefit_criteria = is_benefit_criteria

    def run(self):
        return NotImplemented
