from benchmark.constants import TASKS_ROOT
from benchmark.task.task_model import TaskModelFactory


def main():
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)
    print(task)


if __name__ == '__main__':
    main()
