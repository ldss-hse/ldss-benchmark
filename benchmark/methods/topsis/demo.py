"""
TOPSIS implementation based on https://link.springer.com/book/10.1007/978-3-642-48318-9#about

Ching-Lai Hwang, Kwangsun Yoon. Multiple Attribute Decision Making Methods and Applications.
A State-of-the-Art Survey.
"""
import numpy as np

from benchmark.assessments.decision_matrix import DecisionMatrixFactory, DecisionMatrix
from benchmark.methods.topsis.core import TopsisDecisionMaker


def main():
    print('Running TOPSIS method')
    dm: DecisionMatrix = DecisionMatrixFactory.from_book_aircraft_example()
    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(dm)
    criteria_weights = (.2, .1, .1, .1, .2, .3)
    alternatives_type = (True, True, True, False, True, True)
    decision_maker.set_criteria_weights(criteria_weights)
    decision_maker.set_alternatives_type(alternatives_type)

    decision_maker.run()


if __name__ == '__main__':
    main()
