import enum
from itertools import combinations
from pathlib import Path
from pprint import pprint

import numpy as np

from benchmark.comparison.draw_plots import create_chart, ChartConfig, Language, save_chart
from benchmark.comparison.enums import StatisticsNames
from benchmark.comparison.methods_names import MethodsNames
from benchmark.comparison.schemes.correlation_report_dto import CorrelationReportDTO
from benchmark.comparison.schemes.unique_experimental_setup import UniqueExperimentalSetupDTO
from benchmark.comparison.utils import two_methods_to_key


def load_report_dto(report_path: Path) -> CorrelationReportDTO:
    with report_path.open(encoding='utf-8') as task_file:
        task_raw = task_file.read()
    # pylint: disable=no-member
    return CorrelationReportDTO.__pydantic_model__.parse_raw(task_raw)


def _get_correlation_coefficient(dto: UniqueExperimentalSetupDTO, method_a, method_b, coef_type: StatisticsNames):
    object_to_query = None
    if coef_type is StatisticsNames.KENDALL_TAU:
        object_to_query = dto.kendall_coefficients
    elif coef_type is StatisticsNames.SPEARMAN_RHO:
        object_to_query = dto.spearman_coefficients

    if object_to_query is None:
        return NotImplemented

    coef_value = object_to_query.get(two_methods_to_key(method_a, method_b))
    if coef_value is None:
        coef_value = object_to_query.get(two_methods_to_key(method_b, method_a))
    return round(coef_value, 2)


def compare_two_methods_by_alternatives(dto: CorrelationReportDTO, method_a: MethodsNames, method_b: MethodsNames):
    alternative_to_average = {}
    key_pair = two_methods_to_key(method_a, method_b)
    for alternatives_number in dto.experiment_info.alternatives_range:
        all_configurations_with_this_number = list(filter(
            lambda x: x.setup_info.num_alternatives == alternatives_number,
            dto.unique_configurations
        ))

        for criteria_number in dto.experiment_info.criteria_range:
            all_configurations_with_this_number_and_criteria = list(filter(
                lambda x: x.setup_info.num_criteria == criteria_number,
                all_configurations_with_this_number
            ))

            num_experiments_with_different_criteria = len(all_configurations_with_this_number_and_criteria)
            assert num_experiments_with_different_criteria == 1, \
                'Exactly one unique configuration for each criterion+alternative. Serious fault'
            x = all_configurations_with_this_number_and_criteria[0]

            if alternative_to_average.get(criteria_number) is None:
                alternative_to_average[criteria_number] = {}
            if alternative_to_average[criteria_number].get(alternatives_number) is None:
                alternative_to_average[criteria_number][alternatives_number] = {}
            alternative_to_average[criteria_number][alternatives_number][key_pair] = {
                str(StatisticsNames.KENDALL_TAU): _get_correlation_coefficient(x, method_a, method_b,
                                                                               StatisticsNames.KENDALL_TAU),
                str(StatisticsNames.SPEARMAN_RHO): _get_correlation_coefficient(x, method_a, method_b,
                                                                                StatisticsNames.SPEARMAN_RHO),
            }

    return alternative_to_average


def compare_two_methods_by_criteria(dto, criteria_value: int = None):
    criteria_to_average = {}
    all_pairs = list(combinations((MethodsNames.TOPSIS, MethodsNames.ELECTRE_I, MethodsNames.ML_LDM), 2))
    for method_a, method_b in all_pairs:
        key_pair = two_methods_to_key(method_a, method_b)

        all_configurations_with_this_number = list(filter(
            lambda x: x.setup_info.num_criteria == criteria_value,
            dto.unique_configurations
        ))

        for alternatives_number in dto.experiment_info.alternatives_range:

            all_configurations_with_this_number_and_alternatives = list(filter(
                lambda x: x.setup_info.num_alternatives == alternatives_number,
                all_configurations_with_this_number
            ))

            num_experiments_with_different_alternatives = len(all_configurations_with_this_number_and_alternatives)
            assert num_experiments_with_different_alternatives == 1, \
                'Exactly one unique configuration for each criterion+alternative. Serious fault'
            x = all_configurations_with_this_number_and_alternatives[0]

            if criteria_to_average.get(criteria_value) is None:
                criteria_to_average[criteria_value] = {}
            if criteria_to_average[criteria_value].get(alternatives_number) is None:
                criteria_to_average[criteria_value][alternatives_number] = {}
            criteria_to_average[criteria_value][alternatives_number][key_pair] = {
                str(StatisticsNames.KENDALL_TAU): _get_correlation_coefficient(x, method_a, method_b,
                                                                               StatisticsNames.KENDALL_TAU),
                str(StatisticsNames.SPEARMAN_RHO): _get_correlation_coefficient(x, method_a, method_b,
                                                                                StatisticsNames.SPEARMAN_RHO),
            }

    return criteria_to_average


def extract_for_chart(res_by_criteria, stats_type, criteria_value):
    labels = sorted(res_by_criteria[criteria_value][list(res_by_criteria[criteria_value].keys())[0]].keys())

    rows = {}
    for row_name, row_values in res_by_criteria[criteria_value].items():
        rows[str(row_name)] = np.array([row_values[label][stats_type] for label in labels])

    return labels, rows


def extract_for_pairwise_chart(res_by_alternatives, stats_type, method_a, method_b):
    labels = sorted(
        res_by_alternatives[list(res_by_alternatives.keys())[0]].keys()
    )

    rows = {}
    key_pair = two_methods_to_key(method_a, method_b)
    for row_name, row_values in res_by_alternatives.items():
        acc = []
        for label in labels:
            if row_values[label].get(key_pair) is None:
                key_pair = two_methods_to_key(method_b, method_a)
            acc.append(row_values[label][key_pair][stats_type])
        rows[str(row_name)] = np.array(acc)

    return labels, rows


def save_plot_by_criteria(criteria_level, correlation_report_dto, res_dir_path):
    # key is criteria number, value is mean correlation coefficients for all pairs
    res_by_criteria = compare_two_methods_by_criteria(correlation_report_dto, criteria_value=criteria_level)
    labels, rows = extract_for_chart(res_by_criteria, stats_type=StatisticsNames.KENDALL_TAU,
                                     criteria_value=criteria_level)
    config = ChartConfig(language=Language.ENGLIGH,
                         coefficient_type=StatisticsNames.KENDALL_TAU,
                         font_settings=None,
                         x_title_type='criteria',
                         title=f'Mean of tau for {criteria_level} criteria')
    compiled_plot = create_chart(labels=labels, rows=rows, config=config)

    file_name = f'by_criteria_{criteria_level}_all_methods'
    save_chart(compiled_plot, res_dir_path, file_name, config)


def save_plot_by_alternatives(method_a, method_b, correlation_report_dto, res_dir_path):
    # key is alternatives number, value is mean correlation coefficients for selected two methods
    res_by_alternatives = compare_two_methods_by_alternatives(correlation_report_dto, method_a, method_b)
    labels, rows = extract_for_pairwise_chart(res_by_alternatives, stats_type=StatisticsNames.KENDALL_TAU,
                                              method_a=method_a, method_b=method_b)
    config = ChartConfig(language=Language.ENGLIGH,
                         coefficient_type=StatisticsNames.KENDALL_TAU,
                         font_settings=None,
                         x_title_type='alternatives',
                         title=f'Mean of tau for {method_a} vs {method_b}')
    compiled_plot = create_chart(labels=labels, rows=rows, config=config)

    file_name = f'by_alternatives_{method_a}_vs_{method_b}'
    save_chart(compiled_plot, res_dir_path, file_name, config)


def visualize_correlation_report(full_report_path: Path, res_dir_path: Path):
    correlation_report_dto: CorrelationReportDTO = load_report_dto(full_report_path)

    # 1. Per alternative mode
    for method_a, method_b in combinations((MethodsNames.TOPSIS, MethodsNames.ELECTRE_I, MethodsNames.ML_LDM), 2):
        save_plot_by_alternatives(method_a=method_a,
                                  method_b=method_b,
                                  correlation_report_dto=correlation_report_dto,
                                  res_dir_path=res_dir_path)

    # 2. Per criteria mode
    for criteria_level in correlation_report_dto.experiment_info.criteria_range:
        save_plot_by_criteria(criteria_level=criteria_level,
                              correlation_report_dto=correlation_report_dto,
                              res_dir_path=res_dir_path)
