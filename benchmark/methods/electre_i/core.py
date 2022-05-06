import itertools
from typing import Optional

import networkx as nx
import numpy as np

from benchmark.methods.common.idecision_maker import IDecisionMaker
from benchmark.task.task_model import TaskModel


class ElectreDecisionMaker(IDecisionMaker):
    _concordance: Optional[dict[tuple[int, int], set[int]]]
    _concordance_matrix: Optional[np.ndarray]
    _concordance_dominance_matrix: Optional[np.ndarray]
    _discordance: Optional[dict[tuple[int, int], set[int]]]
    _discordance_matrix: Optional[np.ndarray]
    _discordance_dominance_matrix: Optional[np.ndarray]
    _aggregate_dominance_matrix: Optional[np.ndarray]
    _preference_order_indexes: Optional[np.ndarray]

    def __init__(self, task: TaskModel):
        super().__init__(task)
        self._concordance = None
        self._concordance_matrix = None
        self._discordance = None
        self._discordance_matrix = None
        self._concordance_dominance_matrix = None
        self._discordance_dominance_matrix = None
        self._aggregate_dominance_matrix = None
        self._preference_order_indexes = None

    def run(self):
        # 1. Normalize Decision Matrix
        self._normalize_matrices()

        # 2. Calculate weighted Decision Matrix
        assert self._criteria_weights is not None, 'Decision cannot be made with undefined criteria weights'
        self._apply_criteria_weights()

        # 3. Determine the concordance and discordance sets
        self._determine_concordance_sets()

        # 4. Calculate the concordance matrix
        self._calculate_concordance_matrix()

        # 5. Calculate the discordance matrix
        self._calculate_discordance_matrix()

        # 6. Calculate the concordance dominance matrix
        self._calculate_concordance_dominance_matrix()

        # 7. Calculate the discordance dominance matrix
        self._calculate_discordance_dominance_matrix()

        # 8. Calculate the aggregate dominance matrix
        self._calculate_aggregate_dominance_matrix()

        # 9. Calculate the aggregate dominance matrix
        self._eliminate_alternatives()

        return self._preference_order_indexes

    def _determine_concordance_sets(self):
        self._concordance = {}
        self._discordance = {}

        weighted_dm = self.get_first_decision_matrix().get_weighted()

        alternatives_ids = list(range(len(weighted_dm)))
        comparison_pairs = itertools.product(alternatives_ids, alternatives_ids)
        for k_index, l_index in comparison_pairs:
            if k_index == l_index:
                continue

            k_assessments = weighted_dm[k_index]
            l_assessments = weighted_dm[l_index]

            pair = (k_index, l_index)

            self._concordance[pair] = set()
            self._discordance[pair] = set()

            for criteria_index, (k_assessment, l_assessment) in enumerate(zip(k_assessments, l_assessments)):
                is_benefit_and_ge = k_assessment >= l_assessment and self._criteria_types[criteria_index]
                is_cost_and_le = k_assessment < l_assessment and not self._criteria_types[criteria_index]
                if is_benefit_and_ge or is_cost_and_le:
                    self._concordance[pair].add(criteria_index)
                else:
                    self._discordance[pair].add(criteria_index)

    def _calculate_concordance_matrix(self):
        self._concordance_matrix = np.zeros((self._task.num_alternatives, self._task.num_alternatives))

        for k_index, _ in enumerate(self._concordance_matrix):
            for l_index, _ in enumerate(self._concordance_matrix[k_index]):
                if k_index == l_index:
                    continue
                criteria_indexes = np.array(list(self._concordance[(k_index, l_index)]))
                if criteria_indexes.size == 0:
                    weights = np.array(())
                else:
                    weights = np.array(self._criteria_weights).take(criteria_indexes)
                concordance_index = np.sum(weights) / np.sum(self._criteria_weights)
                self._concordance_matrix[k_index][l_index] = concordance_index

    def _calculate_discordance_matrix(self):
        self._discordance_matrix = np.zeros((self._task.num_alternatives, self._task.num_alternatives))

        decision_matrix = self.get_first_decision_matrix().get_weighted()
        for k_index, _ in enumerate(self._concordance_matrix):
            for l_index, _ in enumerate(self._concordance_matrix[k_index]):
                if k_index == l_index:
                    continue

                discordance_indexes = self._discordance[(k_index, l_index)]
                k_selected_values = decision_matrix[k_index].take(list(discordance_indexes))
                l_selected_values = decision_matrix[l_index].take(list(discordance_indexes))
                if k_selected_values.size == 0 or l_selected_values.size == 0:
                    discordance_max = 0.
                else:
                    discordance_max = np.max(np.abs(k_selected_values - l_selected_values))
                overall_max = np.max(np.abs(decision_matrix[k_index] - decision_matrix[l_index]))

                self._discordance_matrix[k_index][l_index] = discordance_max / overall_max

        self._discordance_matrix = np.around(self._discordance_matrix, 4)

    def _calculate_concordance_dominance_matrix(self):
        _average_concordance_index = np.sum(self._concordance_matrix) / (
                self._task.num_alternatives * (self._task.num_alternatives - 1))
        self._concordance_dominance_matrix = self._concordance_matrix >= _average_concordance_index

    def _calculate_discordance_dominance_matrix(self):
        _average_discordance_index = np.sum(self._discordance_matrix) / (
                    self._task.num_alternatives * (self._task.num_alternatives - 1))
        self._discordance_dominance_matrix = self._discordance_matrix <= _average_discordance_index
        np.fill_diagonal(self._discordance_dominance_matrix, False)

    def _calculate_aggregate_dominance_matrix(self):
        self._aggregate_dominance_matrix = self._concordance_dominance_matrix * self._discordance_dominance_matrix

    def _eliminate_alternatives(self):
        dependency_graph = nx.from_numpy_matrix(self._aggregate_dominance_matrix, create_using=nx.DiGraph)

        # uncomment if you need to visualize the dependency graph
        # import matplotlib.pyplot
        # matplotlib.pyplot.ion()
        # nx.draw(dependency_graph, with_labels=True)

        self._preference_order_indexes = np.array([[i, 0] for i in nx.topological_sort(dependency_graph)])
