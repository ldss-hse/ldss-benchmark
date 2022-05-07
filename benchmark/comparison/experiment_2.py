from benchmark.comparison.experiment_common import run_experiment
from benchmark.comparison.schemes.correlation_report_dto import ExperimentInfoDTO
from benchmark.constants import GENERATED_TASKS_PATH
from benchmark.task.generator.task_type import TaskType


def main():
    experiment_root_path = GENERATED_TASKS_PATH / 'experiment_2'
    experiment_root_path.mkdir(parents=True, exist_ok=True)

    experiment_config = ExperimentInfoDTO(task_type=TaskType.NUMERIC_ONLY,
                                          alternatives_range=(3, ),
                                          criteria_range=(5, ),
                                          num_experts=10,
                                          num_replicas=10,
                                          num_criteria_groups=1,
                                          equal_expert_weights=True)
    run_experiment(experiment_config, experiment_root_path)


if __name__ == '__main__':
    main()
