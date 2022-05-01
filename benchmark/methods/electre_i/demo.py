from benchmark.task.assessments.decision_matrix import DecisionMatrix, DecisionMatrixFactory
from benchmark.methods.electre_i.core import ElectreDecisionMaker


def main():
    print('Running ELECTRE I method')
    decision_matrix: DecisionMatrix = DecisionMatrixFactory.from_book_aircraft_example()
    decision_maker: ElectreDecisionMaker = ElectreDecisionMaker(decision_matrix)
    criteria_weights = (.2, .1, .1, .1, .2, .3)
    alternatives_type = (True, True, True, False, True, True)
    decision_maker.set_criteria_weights(criteria_weights)
    decision_maker.set_alternatives_type(alternatives_type)

    res = decision_maker.run()

    print('ELECTRE I finished making recommendations. In order of decreasing priority:')
    for index, value in res:
        print(f'\tA{int(index)+1}\t{value}')


if __name__ == '__main__':
    main()
