from benchmark.comparison.experiment_common import run_experiment
from benchmark.comparison.schemes.correlation_report_dto import ExperimentInfoDTO
from benchmark.constants import GENERATED_TASKS_PATH
from benchmark.task.generator.task_type import TaskType


def main():
    experiment_root_path = GENERATED_TASKS_PATH / 'experiment_3'
    experiment_root_path.mkdir(parents=True, exist_ok=True)

    experiment_config = ExperimentInfoDTO(task_type=TaskType.HYBRID_CRISP_LINGUISTIC,
                                          alternatives_range=(3, 5, 7, 9),
                                          criteria_range=(5, 10, 15, 20),
                                          num_experts=10,
                                          num_replicas=100,
                                          num_criteria_groups=1,
                                          equal_expert_weights=False,
                                          generate_concrete_expert_weights=False)

    # Additional settings for enabling/disabling first three phases in debug purposes
    experiment_config.generate_new_dataset = False
    experiment_config.execute_decision_makers = False
    experiment_config.calculate_correlation_reports = True

    run_experiment(experiment_config, experiment_root_path)


if __name__ == '__main__':
    main()
