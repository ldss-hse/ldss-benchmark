"""
Comparison follows the methodology explained in:

Blanca Ceballos, Maria T. Lamata, David A. Pelta. A comparative analysis of multi-criteria decision-making methods
doi: 10.1007/s13748-016-0093-1
"""
import shutil

from benchmark.comparison.correlation_report import CorrelationReport
from benchmark.comparison.generate_tasks import generate_tasks
from benchmark.comparison.run_decision_makers import run_decision_makers
from benchmark.comparison.visualize import visualize_correlation_report
from benchmark.constants import GENERATED_TASKS_PATH
from benchmark.task.generator.dumper import save_to_json
from benchmark.task.generator.task_type import TaskType


def main():
    print('Running Experiment no. 1: only numeric assessments, single expert')
    generate_new_dataset = False
    execute_decision_makers = False
    calculate_correlation_reports = False
    task_type = TaskType.NUMERIC_ONLY
    num_replicas = 2

    experiment_root_path = GENERATED_TASKS_PATH / 'experiment_1'
    experiment_reports_path = experiment_root_path / 'reports'
    experiment_tasks_path = experiment_root_path / 'tasks'
    correlation_report_dir_path = experiment_root_path / 'report'
    correlation_report_path = correlation_report_dir_path / 'full_report.json'
    experiment_visualization_path = correlation_report_dir_path / 'visualization'
    experiments_settings_path = experiment_root_path / 'meta.json'
    experiment_reports_path.mkdir(parents=True, exist_ok=True)
    correlation_report_dir_path.mkdir(parents=True, exist_ok=True)
    experiment_visualization_path.mkdir(parents=True, exist_ok=True)

    if generate_new_dataset:
        shutil.rmtree(experiment_tasks_path, ignore_errors=True)
        generate_tasks(experiment_tasks_path, task_type, experiments_settings_path, num_replicas)

    if execute_decision_makers:
        run_decision_makers(experiment_tasks_path, experiment_reports_path)

    if calculate_correlation_reports:
        correlation_report = CorrelationReport(experiment_reports_path, experiments_settings_path)
        correlation_report.build_full_raw_report()
        correlation_report.calculate_correlation()

        save_to_json(correlation_report_path, correlation_report.dto)

    visualize_correlation_report(correlation_report_path, experiment_visualization_path)


if __name__ == '__main__':
    main()
