from benchmark.constants import GENERATED_TASKS_PATH
from benchmark.task.generator.dumper import save_to_json
from benchmark.task.generator.single_task_generator import SingleTaskGenerator
from benchmark.task.generator.task_type import TaskType
from benchmark.task.schemas.task_scheme import TaskDTOScheme


def main():
    generator = SingleTaskGenerator(num_experts=1,
                                    num_alternatives=4,
                                    num_criteria_groups=1,
                                    num_criteria_per_group=6,
                                    task_type=TaskType.NUMERIC_ONLY,
                                    equal_expert_weights=True,
                                    generate_concrete_expert_weights=True)
    res_dto: TaskDTOScheme = generator.run()

    path = GENERATED_TASKS_PATH / 'gen_task_1.json'
    GENERATED_TASKS_PATH.mkdir(exist_ok=True, parents=True)
    save_to_json(path, res_dto)


if __name__ == '__main__':
    main()
