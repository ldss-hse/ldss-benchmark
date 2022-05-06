import enum
from typing import Optional

import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd

from benchmark.comparison.enums import StatisticsNames


def get_img_folder_path():
    project_root = os.path.dirname(os.path.realpath(__file__))
    tmp_artifacts_root = os.path.join(project_root, 'img')
    return tmp_artifacts_root


class Language(str, enum.Enum):
    ENGLIGH = 'ENGLISH'
    RUSSIAN = 'RUSSIAN'

    def __str__(self):
        return self.value


class ChartConfig:
    language: Language
    coefficient_type: StatisticsNames
    font_settings = {
        'fontname': 'Times New Roman',
        'fontsize': 12
    }
    x_title_type: str
    title: str
    file_format: str

    def __init__(self, language: Language, coefficient_type: StatisticsNames, font_settings: Optional[dict],
                 x_title_type: str,
                 title: str, file_format: str = 'png'):
        self.language = language
        self.coefficient_type = coefficient_type
        if font_settings is not None:
            self.font_settings |= font_settings
        self.x_title_type = x_title_type
        self.title = title
        self.file_format = file_format


def create_chart(labels: list[str], rows: dict[str, np.ndarray], config: ChartConfig):
    i8n_mapping = {
        StatisticsNames.SPEARMAN_RHO: {
            Language.RUSSIAN: 'Среднее значение rho',
            Language.ENGLIGH: 'Mean of rho'
        },
        StatisticsNames.KENDALL_TAU: {
            Language.RUSSIAN: 'Среднее значение tau',
            Language.ENGLIGH: 'Mean of tau'
        },
        'alternatives': {
            Language.RUSSIAN: 'Количество альтернатив',
            Language.ENGLIGH: 'Number of alternatives',
        },
        'criteria': {
            Language.RUSSIAN: 'Пары сравниваемых методов',
            Language.ENGLIGH: 'Methods',
        }
    }

    y_title = i8n_mapping[config.coefficient_type][config.language]

    filtered_data_df = pd.DataFrame({
        'x': labels,
        **rows
    })

    plt.close()

    markers = ['o', 's', '^', 'v']
    line_styles = ['solid', 'dotted', 'dashed', 'dashdot']
    for idx, (key, values) in enumerate(rows.items()):
        plt.plot(filtered_data_df['x'], filtered_data_df[key], marker=markers[idx], color='black',
                 linewidth=2, linestyle=line_styles[idx], label=key)

    plt.ylim(-1, 1.2)
    plt.ylabel(y_title, **config.font_settings)
    plt.xlabel(i8n_mapping[config.x_title_type][config.language], **config.font_settings)

    plt.xticks(labels, **config.font_settings)
    plt.yticks(**config.font_settings)

    plt.title(config.title)
    plt.legend()
    # plt.show()

    return plt


def save_chart(plot, artifacts_path, file_name, config: ChartConfig):
    path_template = artifacts_path / f'{file_name}_{str(config.language)}.{config.file_format}'
    plot.savefig(path_template, format=config.file_format, dpi=600)
