import shutil

from benchmark.comparison.correlation_report import CorrelationReport
from benchmark.comparison.draw_plots import Language
from benchmark.comparison.generate_tasks import generate_tasks
from benchmark.comparison.process_report import collect_top_1_matches_from_dto, visualize_correlation_report
from benchmark.comparison.run_decision_makers import run_decision_makers
from benchmark.comparison.schemes.correlation_report_dto import ExperimentInfoDTO
from benchmark.task.generator.dumper import save_to_json


def run_experiment(experiment_config: ExperimentInfoDTO, experiment_root_path):
    print(f'Running new experiment: assessments: {experiment_config.task_type}, {experiment_config.num_experts} expert,'
          f'expert weights are equal: {experiment_config.equal_expert_weights},'
          f'expert weights are assigned automatically: {experiment_config.generate_concrete_expert_weights}')

    experiment_reports_path = experiment_root_path / 'reports'
    experiment_tasks_path = experiment_root_path / 'tasks'
    correlation_report_dir_path = experiment_root_path / 'report'
    correlation_report_path = correlation_report_dir_path / 'full_report.json'
    experiment_visualization_path = correlation_report_dir_path / 'visualization'
    experiments_settings_path = experiment_root_path / 'meta.json'
    experiment_reports_path.mkdir(parents=True, exist_ok=True)
    correlation_report_dir_path.mkdir(parents=True, exist_ok=True)
    experiment_visualization_path.mkdir(parents=True, exist_ok=True)

    if experiment_config.generate_new_dataset:
        shutil.rmtree(experiment_tasks_path, ignore_errors=True)
        generate_tasks(experiment_tasks_path, experiments_settings_path, experiment_config, is_multiprocessing=True)

    if experiment_config.execute_decision_makers:
        run_decision_makers(experiment_tasks_path, experiment_reports_path, is_multiprocessing=True)

    if experiment_config.calculate_correlation_reports:
        correlation_report = CorrelationReport(experiment_reports_path, experiments_settings_path)
        correlation_report.build_full_raw_report()
        correlation_report.calculate_correlation()

        save_to_json(correlation_report_path, correlation_report.dto)

    collect_top_1_matches_from_dto(correlation_report_path, res_dir_path=correlation_report_dir_path)

    visualize_correlation_report(correlation_report_path, experiment_visualization_path, language=Language.ENGLIGH)
