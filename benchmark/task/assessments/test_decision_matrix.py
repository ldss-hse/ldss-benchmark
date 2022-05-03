import numpy as np

from benchmark.constants import TASKS_ROOT
from benchmark.task.task_model import TaskModelFactory


def test_matrix_normalization():
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)
    decision_matrix = list(task._decision_matrices.values())[0]

    decision_matrix.normalize()

    expected_matrix_raw = [
        [.4671, .3662, .5056, .5069, .4811, .6708],
        [.5839, .6591, .4550, .5990, .2887, .3727],
        [.4204, .4882, .5308, .4147, .6736, .5217],
        [.5139, .4394, .5056, .4608, .4811, .3727],
    ]
    expected_matrix = np.array(expected_matrix_raw)

    assert np.array_equal(decision_matrix.get_normalized(), expected_matrix), 'Normalization failed'


def test_criteria_weighting():
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)
    decision_matrix = list(task._decision_matrices.values())[0]

    decision_matrix.normalize()
    criteria_weights = (.2, .1, .1, .1, .2, .3)
    decision_matrix.apply_criteria_weights(criteria_weights)

    expected_matrix_raw = [
        [.0934, .0366, .0506, .0507, .0962, .2012],
        [.1168, .0659, .0455, .0599, .0577, .1118],
        [.0841, .0488, .0531, .0415, .1347, .1565],
        [.1028, .0439, .0506, .0461, .0962, .1118],
    ]
    expected_matrix = np.array(expected_matrix_raw)

    assert np.array_equal(decision_matrix.get_weighted(), expected_matrix), 'Weighting failed'


def test_criteria_weighting_from_book():
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)
    decision_matrix = list(task._decision_matrices.values())[0]

    # in the original book example seems to contain errors during normalization. However,
    # in order to follow other computations we need to keep their normalized matrix
    their_normalized = [
        [.4671, .3662, .5056, .5063, .4811, .6708],
        [.5839, .6591, .4550, .5983, .2887, .3727],
        [.4204, .4882, .5308, .4143, .6736, .5217],
        [.5139, .4392, .5056, .4603, .4811, .3727],
    ]
    decision_matrix._normalized = np.array(their_normalized)

    criteria_weights = (.2, .1, .1, .1, .2, .3)
    decision_matrix.apply_criteria_weights(criteria_weights)

    expected_matrix_weighted = [
        [.0934, .0366, .0506, .0506, .0962, .2012],
        [.1168, .0659, .0455, .0598, .0577, .1118],
        [.0841, .0488, .0531, .0414, .1347, .1565],
        [.1028, .0439, .0506, .0460, .0962, .1118],
    ]
    expected_matrix = np.array(expected_matrix_weighted)

    assert np.array_equal(decision_matrix.get_weighted(), expected_matrix), 'Weighting failed'
