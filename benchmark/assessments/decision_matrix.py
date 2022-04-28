import math
from typing import Optional

import numpy as np


class DecisionMatrix:
    _raw: np.ndarray
    _normalized: Optional[np.ndarray]
    _weighted: Optional[np.ndarray]

    def __init__(self, raw_assessments: np.ndarray):
        self._raw = raw_assessments
        self._normalized = None
        self._weighted = None

    def get_normalized(self):
        return self._normalized

    def get_weighted(self) -> np.ndarray:
        return self._weighted

    def normalize(self):
        if self._normalized is not None:
            return
        dividers = np.apply_along_axis(lambda column: math.sqrt((column**2).sum()), 0, self._raw)
        self._normalized = np.around(self._raw / dividers, decimals=4)

    def apply_criteria_weights(self, criteria_weights):
        to_multiply = self._raw if self._normalized is None else self._normalized
        self._weighted = np.around(to_multiply * np.array(criteria_weights), decimals=4)


class DecisionMatrixFactory:
    @staticmethod
    def from_book_aircraft_example() -> DecisionMatrix:
        """
        Example taken from TOPSIS book, page 133
        :return: DecisionMatrix instance with raw assessments
        """
        raw_data = [
            [2., 1500, 20000, 5.5, 5, 9],
            [2.5, 2700, 18000, 6.5, 3, 5],
            [1.8, 2000, 21000, 4.5, 7, 7],
            [2.2, 1800, 20000, 5.0, 5, 5],
        ]
        return DecisionMatrix(np.array(raw_data))
