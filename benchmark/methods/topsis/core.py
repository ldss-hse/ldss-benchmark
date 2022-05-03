from typing import Optional

import numpy as np

from benchmark.methods.common.idecision_maker import IDecisionMaker
from benchmark.task.task_model import TaskModel


class TopsisDecisionMaker(IDecisionMaker):
    _artificial_ideal_alternative: Optional[np.ndarray]
    _artificial_ideal_negative_alternative: Optional[np.ndarray]
    _separation_measure_from_ideal: Optional[np.ndarray]
    _separation_measure_from_ideal_negative: Optional[np.ndarray]
    _relative_closeness: Optional[np.ndarray]
    _preference_order_indexes: Optional[np.ndarray]

    def __init__(self, task: TaskModel):
        super().__init__(task)
        self.is_benefit_criteria = None
        self._artificial_ideal_alternative = None
        self._artificial_ideal_negative_alternative = None

    def run(self):
        # 1. Normalize Decision Matrix
        self._normalize_matrices()

        # 2. Calculate weighted Decision Matrix
        assert self._criteria_weights is not None, 'Decision cannot be made with undefined criteria weights'
        self._apply_criteria_weights()

        # 3. Determine ideal and negative-ideal solutions
        assert self._criteria_types is not None, 'Decision cannot be made with undefined criteria types'
        self._define_artificial_ideal_alternatives()

        # 4. Calculate separation measures
        self._define_separation_measures()

        # 5. Calculate relative closeness to the ideal solution
        self._calculate_relative_closeness()

        # 6. Rank the preference order
        self._rank_preference_order()

        indexes = self._preference_order_indexes
        values = self._relative_closeness[self._preference_order_indexes]
        return np.column_stack((indexes, values))[::-1, :]

    def _define_artificial_ideal_alternatives(self):
        decision_matrix = self.get_first_decision_matrix()
        all_maximums = np.amax(decision_matrix.get_weighted(), axis=0)
        all_minimums = np.amin(decision_matrix.get_weighted(), axis=0)

        ideal_alternative = []
        for index, is_benefit in enumerate(self._criteria_types):
            base_array = all_maximums if is_benefit else all_minimums
            ideal_alternative.append(base_array[index])

        self._artificial_ideal_alternative = np.array(ideal_alternative)

        ideal_negative_alternative = []
        for index, is_benefit in enumerate(self._criteria_types):
            base_array = all_minimums if is_benefit else all_maximums
            ideal_negative_alternative.append(base_array[index])

        self._artificial_ideal_negative_alternative = np.array(ideal_negative_alternative)

    def _define_separation_measures(self):
        decision_matrix = self.get_first_decision_matrix()
        weighted_dm = decision_matrix.get_weighted()

        def calculate_distance(matrix, vector):
            return np.around(np.sqrt(np.sum((matrix - vector) ** 2, axis=1)), decimals=4)

        self._separation_measure_from_ideal = calculate_distance(weighted_dm, self._artificial_ideal_alternative)
        self._separation_measure_from_ideal_negative = calculate_distance(weighted_dm,
                                                                          self._artificial_ideal_negative_alternative)

    def _calculate_relative_closeness(self):
        s_i_minus = self._separation_measure_from_ideal_negative
        s_i_star = self._separation_measure_from_ideal

        c_i_star = s_i_minus / (s_i_star + s_i_minus)
        self._relative_closeness = np.around(c_i_star, decimals=3)

    def _rank_preference_order(self):
        self._preference_order_indexes = np.argsort(self._relative_closeness)
