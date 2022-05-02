"""
TOPSIS implementation based on https://link.springer.com/book/10.1007/978-3-642-48318-9#about

Ching-Lai Hwang, Kwangsun Yoon. Multiple Attribute Decision Making Methods and Applications.
A State-of-the-Art Survey.
"""
from benchmark.constants import TASKS_ROOT
from benchmark.task.assessments.decision_matrix import DecisionMatrixFactory, DecisionMatrix
from benchmark.methods.topsis.core import TopsisDecisionMaker
from benchmark.task.task_model import TaskModelFactory


def main():
    print('Running TOPSIS method')
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)

    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(task)

    res = decision_maker.run()

    print('TOPSIS finished making recommendations. In order of decreasing priority:')
    for index, value in res:
        print(f'\tA{int(index)+1}\t{value}')


if __name__ == '__main__':
    main()
