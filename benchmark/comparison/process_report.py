import enum
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.comparison.draw_plots import create_chart, ChartConfig, Language, save_chart
from benchmark.comparison.enums import StatisticsNames
from benchmark.comparison.methods_names import MethodsNames
from benchmark.comparison.schemes.correlation_report_dto import CorrelationReportDTO
from benchmark.comparison.utils import two_methods_to_key


class DFReportColumnsNames(str, enum.Enum):
    NUM_CRITERIA = 'num_criteria'
    NUM_ALTERNATIVES = 'num_alternatives'
    NUM_EXPERTS = 'num_experts'
    NUM_CRITERIA_GROUPS = 'num_criteria_groups'
    METHODS_NAME = 'method'
    COEFFICIENT_TYPE = 'coefficient_type'
    COEFFICIENT_VALUE = 'coefficient_value'

    def __str__(self):
        return self.value


def load_report_dto(report_path: Path) -> CorrelationReportDTO:
    with report_path.open(encoding='utf-8') as task_file:
        task_raw = task_file.read()
    # pylint: disable=no-member
    return CorrelationReportDTO.__pydantic_model__.parse_raw(task_raw)


def create_df_report_from_dto(dto: CorrelationReportDTO) -> pd.DataFrame:
    all_rows = []
    for unique_configuration in dto.unique_configurations:
        sample_row = {
            str(DFReportColumnsNames.NUM_CRITERIA): unique_configuration.setup_info.num_criteria,
            str(DFReportColumnsNames.NUM_ALTERNATIVES): unique_configuration.setup_info.num_alternatives,
            str(DFReportColumnsNames.NUM_EXPERTS): unique_configuration.setup_info.num_experts,
            str(DFReportColumnsNames.NUM_CRITERIA_GROUPS): unique_configuration.setup_info.num_criteria_groups,

        }
        for stats_type, actual_coefficients in zip((StatisticsNames.KENDALL_TAU, StatisticsNames.SPEARMAN_RHO),
                                                   (unique_configuration.kendall_coefficients,
                                                    unique_configuration.spearman_coefficients)):
            for methods_pair, value in actual_coefficients.items():
                all_rows.append({
                    **sample_row,
                    str(DFReportColumnsNames.METHODS_NAME): methods_pair,
                    str(DFReportColumnsNames.COEFFICIENT_TYPE): stats_type,
                    str(DFReportColumnsNames.COEFFICIENT_VALUE): float(value)
                })

    experiment_df = pd.DataFrame(all_rows)
    assert len(experiment_df['method'].unique()) == 3, 'Only three methods are considered to this moment'

    return experiment_df


def save_plot_by_criteria(criteria_level, stats_type: StatisticsNames, correlation_report_df: pd.DataFrame,
                          res_dir_path, language: Language):
    filtered_df = correlation_report_df[
        (correlation_report_df[str(DFReportColumnsNames.NUM_CRITERIA)] == criteria_level) &
        (correlation_report_df[str(DFReportColumnsNames.COEFFICIENT_TYPE)] == str(stats_type))
        ]

    labels = list(correlation_report_df[str(DFReportColumnsNames.METHODS_NAME)].unique())

    unique_alternatives = list(correlation_report_df[str(DFReportColumnsNames.NUM_ALTERNATIVES)].unique())
    rows = {
        alternative_number: np.zeros((len(labels))) for alternative_number in unique_alternatives
    }
    for alternative_number in unique_alternatives:
        for method_index, method_pair in enumerate(labels):
            value = filtered_df[
                (filtered_df[str(DFReportColumnsNames.METHODS_NAME)] == method_pair) &
                (filtered_df[str(DFReportColumnsNames.NUM_ALTERNATIVES)] == alternative_number)
                ]
            assert len(value[str(DFReportColumnsNames.COEFFICIENT_VALUE)]) == 1, 'Only one experiment can occur with' \
                                                                                 'such settings. Serious error'
            rows[alternative_number][method_index] = value[str(DFReportColumnsNames.COEFFICIENT_VALUE)].values[0]

    config = ChartConfig(language=language,
                         coefficient_type=stats_type,
                         font_settings=None,
                         x_title_type='criteria')
    compiled_plot = create_chart(labels=labels, rows=rows, config=config)

    file_name = f'by_criteria_{criteria_level}_all_methods_{stats_type}'
    save_chart(compiled_plot, res_dir_path, file_name, config)


def save_plot_by_alternatives(method_a, method_b, stats_type: StatisticsNames, correlation_report_df: pd.DataFrame,
                              res_dir_path, language: Language):
    methods_pairs = list(correlation_report_df[str(DFReportColumnsNames.METHODS_NAME)].unique())

    key_pair = two_methods_to_key(method_a, method_b)
    if key_pair not in methods_pairs:
        key_pair = two_methods_to_key(method_b, method_a)

    filtered_df = correlation_report_df[
        (correlation_report_df[str(DFReportColumnsNames.METHODS_NAME)] == key_pair) &
        (correlation_report_df[str(DFReportColumnsNames.COEFFICIENT_TYPE)] == str(stats_type))
        ]

    labels = list(correlation_report_df[str(DFReportColumnsNames.NUM_ALTERNATIVES)].unique())
    unique_criteria = list(correlation_report_df[str(DFReportColumnsNames.NUM_CRITERIA)].unique())
    rows = {
        criteria_number: np.zeros((len(labels))) for criteria_number in unique_criteria
    }

    for criteria_number in unique_criteria:
        for alternative_index, alternative_number in enumerate(labels):
            value = filtered_df[
                (filtered_df[str(DFReportColumnsNames.NUM_CRITERIA)] == criteria_number) &
                (filtered_df[str(DFReportColumnsNames.NUM_ALTERNATIVES)] == alternative_number)
                ]
            assert len(value[str(DFReportColumnsNames.COEFFICIENT_VALUE)]) == 1, 'Only one experiment can occur with' \
                                                                                 'such settings. Serious error'
            rows[criteria_number][alternative_index] = value[str(DFReportColumnsNames.COEFFICIENT_VALUE)].values[0]

    config = ChartConfig(language=language,
                         coefficient_type=stats_type,
                         font_settings=None,
                         x_title_type='alternatives')
    compiled_plot = create_chart(labels=labels, rows=rows, config=config)

    file_name = f'by_alternatives_{method_a}_vs_{method_b}_{stats_type}'
    save_chart(compiled_plot, res_dir_path, file_name, config)


def save_averages_pairwise_chart(correlation_report_df: pd.DataFrame, stats_type, res_dir_path, language: Language):
    filtered_df = correlation_report_df[
        correlation_report_df[str(DFReportColumnsNames.COEFFICIENT_TYPE)] == str(stats_type)]
    grouped_df = filtered_df \
        .groupby(
        [
            str(DFReportColumnsNames.METHODS_NAME),
            str(DFReportColumnsNames.NUM_ALTERNATIVES)
        ]) \
        .agg({str(DFReportColumnsNames.COEFFICIENT_VALUE): ['mean']})

    labels = list(correlation_report_df[str(DFReportColumnsNames.METHODS_NAME)].unique())

    unique_alternatives = list(correlation_report_df[str(DFReportColumnsNames.NUM_ALTERNATIVES)].unique())
    rows = {
        alternative_number: np.zeros((len(labels))) for alternative_number in unique_alternatives
    }
    for name in grouped_df.index:
        methods_pair, alternative_number = name
        rows[alternative_number][labels.index(methods_pair)] = grouped_df.loc[name][0]

    config = ChartConfig(language=language,
                         coefficient_type=stats_type,
                         font_settings=None,
                         x_title_type='alternatives')
    compiled_plot = create_chart(labels=labels, rows=rows, config=config)

    file_name = f'aggregated_all_methods_{stats_type}'
    save_chart(compiled_plot, res_dir_path, file_name, config)


def visualize_correlation_report(full_report_path: Path, res_dir_path: Path, language: Language = Language.RUSSIAN):
    correlation_report_dto: CorrelationReportDTO = load_report_dto(full_report_path)

    correlation_report_df: pd.DataFrame = create_df_report_from_dto(correlation_report_dto)

    for stats_type in (StatisticsNames.KENDALL_TAU, StatisticsNames.SPEARMAN_RHO):
        save_averages_pairwise_chart(correlation_report_df, stats_type=stats_type, res_dir_path=res_dir_path,
                                     language=language)

    all_pairs = list(combinations((MethodsNames.TOPSIS, MethodsNames.ELECTRE_I, MethodsNames.ML_LDM), 2))

    for stats_type in (StatisticsNames.KENDALL_TAU, StatisticsNames.SPEARMAN_RHO):
        # 1. Per alternative mode
        for method_a, method_b in all_pairs:
            save_plot_by_alternatives(method_a=method_a,
                                      method_b=method_b,
                                      stats_type=stats_type,
                                      correlation_report_df=correlation_report_df,
                                      res_dir_path=res_dir_path,
                                      language=language)

        # 2. Per criteria mode
        for criteria_level in correlation_report_dto.experiment_info.criteria_range:
            save_plot_by_criteria(criteria_level=criteria_level,
                                  stats_type=stats_type,
                                  correlation_report_df=correlation_report_df,
                                  res_dir_path=res_dir_path,
                                  language=language)


def collect_top_1_matches_from_dto(full_report_path: Path, res_dir_path: Path):
    correlation_report_dto: CorrelationReportDTO = load_report_dto(full_report_path)

    top_1_hits = {}
    total_quantity = 0
    for unique_configuration in correlation_report_dto.unique_configurations:
        for methods_pair, quantity in unique_configuration.top_1_matches.items():
            top_1_hits[methods_pair] = top_1_hits.get(methods_pair, 0) + quantity
        total_quantity += unique_configuration.total_runs

    rows = []
    for methods_pair, hits in top_1_hits.items():
        rows.append({
            'method': methods_pair,
            'hits': hits / total_quantity
        })
    df = pd.DataFrame(rows)
    df.to_csv(res_dir_path / 'top_1_hits.tsv', sep='\t')
