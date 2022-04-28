import itertools
from typing import Optional

import numpy as np

from benchmark.assessments.decision_matrix import DecisionMatrix
from benchmark.methods.common.idecision_maker import IDecisionMaker


class ElectreDecisionMaker(IDecisionMaker):
    _concordance: Optional[dict[tuple[int, int], set[int]]]
    _discordance: Optional[dict[tuple[int, int], set[int]]]

    def __init__(self, decision_matrix: DecisionMatrix):
        super().__init__(decision_matrix)
        self._concordance = None
        self._discordance = None

    def run(self):
        # 1. Normalize Decision Matrix
        self._decision_matrix.normalize()

        # 2. Calculate weighted Decision Matrix
        assert self.criteria_weights is not None, 'Decision cannot be made with undefined criteria weights'
        self._decision_matrix.apply_criteria_weights(self.criteria_weights)

        # 3. Determine the concordance and discordance sets
        self._determine_concordance_sets()

    def _determine_concordance_sets(self):
        self._concordance = {}
        self._discordance = {}

        weighted_dm = self._decision_matrix.get_weighted()
        comparison_pairs = itertools.combinations(list(range(len(weighted_dm))), 2)
        for k_index, l_index in comparison_pairs:
            concordance_indexes = set(np.argwhere(weighted_dm[k_index] >= weighted_dm[l_index]).flatten())
            self._concordance[(k_index, l_index)] = concordance_indexes

            discordance_indexes = set(np.argwhere(weighted_dm[k_index] < weighted_dm[l_index]).flatten())
            self._discordance[(k_index, l_index)] = discordance_indexes


