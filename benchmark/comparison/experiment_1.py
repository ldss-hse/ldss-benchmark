"""
Comparison follows the methodology explained in:

Blanca Ceballos, Maria T. Lamata, David A. Pelta. A comparative analysis of multi-criteria decision-making methods
doi: 10.1007/s13748-016-0093-1
"""
from benchmark.comparison.experiment_common import run_experiment
from benchmark.comparison.schemes.correlation_report_dto import ExperimentInfoDTO
from benchmark.constants import GENERATED_TASKS_PATH
from benchmark.task.generator.task_type import TaskType


def main():
    experiment_root_path = GENERATED_TASKS_PATH / 'experiment_1'
    experiment_root_path.mkdir(parents=True, exist_ok=True)

    experiment_config = ExperimentInfoDTO(task_type=TaskType.NUMERIC_ONLY,
                                          alternatives_range=(3, 5, 7, 9),
                                          criteria_range=(5, 10, 15, 20),
                                          num_experts=1,
                                          num_replicas=100,
                                          num_criteria_groups=1,
                                          equal_expert_weights=True,
                                          generate_concrete_expert_weights=True)
    run_experiment(experiment_config, experiment_root_path)


if __name__ == '__main__':
    main()
