import enum
from itertools import combinations
from pathlib import Path
from pprint import pprint

import numpy as np

from benchmark.comparison.draw_plots import create_chart
from benchmark.comparison.methods_names import MethodsNames
from benchmark.comparison.schemes.correlation_report_dto import CorrelationReportDTO
from benchmark.comparison.schemes.unique_experimental_setup import UniqueExperimentalSetupDTO
from benchmark.comparison.utils import two_methods_to_key


class StatisticsNames(str, enum.Enum):
    KENDALL_TAU = 'KENDALL_TAU'
    SPEARMAN_RHO = 'SPEARMAN_RHO'

    def __str__(self):
        return self.value


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
    for alternatives_number in dto.experiment_info.alternatives_range:
        all_configurations_with_this_number = list(filter(
            lambda x: x.setup_info.num_alternatives == alternatives_number,
            dto.unique_configurations
        ))
        kendall_tau_values = list(map(
            lambda x: _get_correlation_coefficient(x, method_a, method_b, StatisticsNames.KENDALL_TAU),
            all_configurations_with_this_number))
        spearman_rho_values = list(map(
            lambda x: _get_correlation_coefficient(x, method_a, method_b, StatisticsNames.SPEARMAN_RHO),
            all_configurations_with_this_number))

        num_experiments_with_different_criteria = len(list(all_configurations_with_this_number))
        assert num_experiments_with_different_criteria == len(dto.experiment_info.criteria_range), \
            'Missed some criteria experiments. Serious fault'

        alternative_to_average[alternatives_number] = {
            str(StatisticsNames.KENDALL_TAU): round(sum(kendall_tau_values) / num_experiments_with_different_criteria,
                                                    2),
            str(StatisticsNames.SPEARMAN_RHO): round(sum(spearman_rho_values) / num_experiments_with_different_criteria,
                                                     2),
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


def visualize_correlation_report(full_report_path: Path, res_dir_path: Path):
    correlation_report_dto: CorrelationReportDTO = load_report_dto(full_report_path)

    method_a = MethodsNames.TOPSIS
    method_b = MethodsNames.ELECTRE_I

    # 1. Per alternative mode
    # key is alternatives number, value is mean correlation coefficients for selected two methods
    res_by_alternatives = compare_two_methods_by_alternatives(correlation_report_dto, method_a, method_b)

    # 2. Per criteria mode
    # key is criteria number, value is mean correlation coefficients for all pairs
    res_by_criteria = compare_two_methods_by_criteria(correlation_report_dto, criteria_value=5)
    labels, rows = extract_for_chart(res_by_criteria, stats_type=StatisticsNames.KENDALL_TAU, criteria_value=5)
    pprint(labels)
    pprint(rows)
    # create_chart(data=res_by_criteria)
