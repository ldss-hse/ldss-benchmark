from benchmark.constants import TASKS_ROOT
from benchmark.methods.electre_i.core import ElectreDecisionMaker
from benchmark.task.task_model import TaskModelFactory


def main():
    print('Running ELECTRE I method')
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)

    decision_maker: ElectreDecisionMaker = ElectreDecisionMaker(task)

    res = decision_maker.run()

    print('ELECTRE I finished making recommendations. In order of decreasing priority:')
    for index, value in res:
        print(f'\tA{int(index)+1}\t{value}')


if __name__ == '__main__':
    main()
