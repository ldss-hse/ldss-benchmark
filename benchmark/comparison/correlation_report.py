import json
from itertools import combinations
from pathlib import Path
from typing import Optional

from scipy.stats import kendalltau, spearmanr

from benchmark.comparison.methods_names import MethodsNames
from benchmark.comparison.schemes.correlation_report_dto import CorrelationReportDTO, ExperimentInfoDTO
from benchmark.comparison.schemes.unique_experimental_setup import UniqueExperimentalSetupDTO, _create_from_json
from benchmark.comparison.utils import two_methods_to_key
from benchmark.task.generator.task_type import TaskType


class UniqueExperimentalSetup:
    _dto: Optional[UniqueExperimentalSetupDTO]
    _data: dict[MethodsNames, list[list[int]]]

    def __init__(self, info):
        self._dto = UniqueExperimentalSetupDTO(setup_info=_create_from_json(info),
                                               kendall_coefficients={},
                                               spearman_coefficients={},
                                               top_1_matches={})
        self._data = {}

    def add_new_data(self, method_name: MethodsNames, param: list[int]):
        if self._data.get(method_name) is None:
            self._data[method_name] = []
        self._data[method_name].append(param)

    def compare_pairwise(self):
        pairs = combinations((MethodsNames.ELECTRE_I, MethodsNames.TOPSIS, MethodsNames.ML_LDM), 2)
        for method_a, method_b in pairs:
            kendall_coefficients = []
            spearman_coefficients = []

            self._dto.total_runs = len(self._data[method_a])

            top_1_hits = 0
            for ranks_a, ranks_b in zip(self._data[method_a], self._data[method_b]):
                kendall_coefficients.append(kendalltau(ranks_a, ranks_b)[0])
                spearman_coefficients.append(spearmanr(ranks_a, ranks_b)[0])

                if ranks_a[0] == ranks_b[0]:
                    top_1_hits += 1

            total_elements = len(kendall_coefficients)
            key = two_methods_to_key(method_a, method_b)
            self._dto.kendall_coefficients[key] = sum(kendall_coefficients) / total_elements
            self._dto.spearman_coefficients[key] = sum(spearman_coefficients) / total_elements

            self._dto.top_1_matches[key] = top_1_hits

    @property
    def dto(self):
        return self._dto


def _load_experiments_info(experiments_configuration_path):
    with experiments_configuration_path.open(encoding='utf-8') as file:
        data = json.load(file)

    return ExperimentInfoDTO(
        task_type=TaskType(data['task_type']),
        alternatives_range=data['alternatives_range'],
        criteria_range=data['criteria_range'],
        num_experts=data['num_experts'],
        num_criteria_groups=data['num_criteria_groups'],
        num_replicas=data.get('num_replicas')
    )


class CorrelationReport:
    _dto: CorrelationReportDTO
    _unique_combinations: list[UniqueExperimentalSetup]
    _report_directory_path: Path

    def __init__(self, experiment_reports_path, experiments_configuration_path: Path):
        self._dto = CorrelationReportDTO(unique_configurations=[],
                                         experiment_info=_load_experiments_info(experiments_configuration_path))
        self._unique_combinations = []
        self._report_directory_path = experiment_reports_path

    @property
    def unique_combinations(self):
        return self._unique_combinations

    @property
    def dto(self):
        return self._dto

    def find_existing_setup(self, json_dict: dict):
        founds = list(filter(lambda x: x.dto.setup_info == json_dict['task_info'], self._unique_combinations))
        if founds:
            return founds[0]
        return None

    def add_new_setup(self, json_dict):
        if self.find_existing_setup(json_dict):
            return None
        new_setup = UniqueExperimentalSetup(json_dict['task_info'])
        self._unique_combinations.append(new_setup)
        return new_setup

    def build_full_raw_report(self):
        all_files = sorted(self._report_directory_path.glob('*.json'), key=lambda f: int(f.stem.split('_')[-1]))
        num_files = len(all_files)
        for file_idx, file_path in enumerate(all_files):
            print(f'{file_idx + 1}/{num_files} Loading report for correlation analysis from {file_path}...', end=' ')

            with file_path.open(encoding='utf-8') as json_file:
                data = json.load(json_file)

            if data.get('failed'):
                print('skip due to error.')
                continue

            unique_setup: UniqueExperimentalSetup = self.find_existing_setup(data)
            if not unique_setup:
                unique_setup = self.add_new_setup(data)

            unique_setup.add_new_data(MethodsNames.ML_LDM, data[str(MethodsNames.ML_LDM)])
            unique_setup.add_new_data(MethodsNames.ELECTRE_I, data[str(MethodsNames.ELECTRE_I)])
            unique_setup.add_new_data(MethodsNames.TOPSIS, data[str(MethodsNames.TOPSIS)])

            print('done.')

    def calculate_correlation(self):
        for unique_combination in self._unique_combinations:
            unique_combination: UniqueExperimentalSetup
            unique_combination.compare_pairwise()
            self._dto.unique_configurations.append(unique_combination.dto)
