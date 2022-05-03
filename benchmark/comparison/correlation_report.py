import json
from itertools import combinations
from pathlib import Path

from scipy.stats import kendalltau, spearmanr

from benchmark.comparison.methods_names import MethodsNames


class UniqueExperimentalSetup:
    _setup_info: dict[str, int]
    _data: dict[MethodsNames, list[list[int]]]
    _kendall_coefficients: dict[(str, str), list[float]]
    _spearman_coefficients: dict[(str, str), list[float]]

    def __init__(self, info):
        self._setup_info = info
        self._data = {}
        self._kendall_coefficients = {}
        self._spearman_coefficients = {}

    def add_new_data(self, method_name: MethodsNames, param: list[int]):
        if self._data.get(method_name) is None:
            self._data[method_name] = []
        self._data[method_name].append(param)

    def compare_pairwise(self):
        pairs = combinations((MethodsNames.ELECTRE_I, MethodsNames.TOPSIS, MethodsNames.ML_LDM), 2)
        for method_a, method_b in pairs:
            kendall_coefficients = []
            spearman_coefficients = []
            for ranks_a, ranks_b in zip(self._data[method_a], self._data[method_b]):
                kendall_coefficients.append(kendalltau(ranks_a, ranks_b)[0])
                spearman_coefficients.append(spearmanr(ranks_a, ranks_b)[0])
            self._kendall_coefficients[(method_a, method_b)] = sum(kendall_coefficients) / len(kendall_coefficients)
            self._spearman_coefficients[(method_a, method_b)] = sum(spearman_coefficients) / len(spearman_coefficients)

    def __str__(self):
        kendall_pieces = []
        for (method_a, method_b), value in self._kendall_coefficients.items():
            kendall_pieces.append(f'{method_a} vs {method_b} :\t{value: .3f}')
        kendall_piece = "\n\t\t".join(kendall_pieces)
        spearman_pieces = []
        for (method_a, method_b), value in self._spearman_coefficients.items():
            spearman_pieces.append(f'{method_a} vs {method_b} :\t{value: .3f}')
        spearman_piece = "\n\t\t".join(spearman_pieces)
        return f'Setup:' \
               f'\n\tExperts: {self._setup_info["num_experts"]} ' \
               f'\n\tAlternatives: {self._setup_info["num_alternatives"]}, ' \
               f'\n\tCriteria: {self._setup_info["num_criteria"]}' \
               f'\n\tAverage Kendell: {kendall_piece}' \
               f'\n\tAverage Spearman: {spearman_piece}'


class CorrelationReport:
    _unique_combinations: list[UniqueExperimentalSetup]
    _report_directory_path: Path

    def __init__(self, experiment_reports_path):
        self._unique_combinations = []
        self._report_directory_path = experiment_reports_path

    @property
    def unique_combinations(self):
        return self._unique_combinations

    def find_existing_setup(self, json_dict: dict):
        founds = list(filter(lambda x: x._setup_info == json_dict['task_info'], self._unique_combinations))
        if founds:
            return founds[0]
        return None

    def add_new_setup(self, json_dict):
        if self.find_existing_setup(json_dict):
            return
        new_setup = UniqueExperimentalSetup(json_dict['task_info'])
        self._unique_combinations.append(new_setup)
        return new_setup

    def build_full_raw_report(self):
        all_files = sorted(self._report_directory_path.glob('*.json'), key=lambda f: int(f.stem.split('_')[-1]))
        num_files = len(all_files)
        for file_idx, file_path in enumerate(all_files):
            print(f'{file_idx + 1}/{num_files} Processing {file_path}...', end=' ')

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

    def calculate_correlation(self):
        for unique_combination in self._unique_combinations:
            unique_combination: UniqueExperimentalSetup
            unique_combination.compare_pairwise()
