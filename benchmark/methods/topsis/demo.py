"""
TOPSIS implementation based on https://link.springer.com/book/10.1007/978-3-642-48318-9#about

Ching-Lai Hwang, Kwangsun Yoon. Multiple Attribute Decision Making Methods and Applications.
A State-of-the-Art Survey.
"""

from benchmark.task.assessments.decision_matrix import DecisionMatrixFactory, DecisionMatrix
from benchmark.methods.topsis.core import TopsisDecisionMaker


def main():
    print('Running TOPSIS method')
    decision_matrix: DecisionMatrix = DecisionMatrixFactory.from_book_aircraft_example()
    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(decision_matrix)
    criteria_weights = (.2, .1, .1, .1, .2, .3)
    alternatives_type = (True, True, True, False, True, True)
    decision_maker.set_criteria_weights(criteria_weights)
    decision_maker.set_alternatives_type(alternatives_type)

    res = decision_maker.run()

    print('TOPSIS finished making recommendations. In order of decreasing priority:')
    for index, value in res:
        print(f'\tA{int(index)+1}\t{value}')


if __name__ == '__main__':
    main()
