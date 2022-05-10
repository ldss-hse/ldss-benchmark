import numpy as np

from benchmark.constants import TASKS_ROOT
from benchmark.methods.topsis.core import TopsisDecisionMaker
from benchmark.task.task_model import TaskModelFactory


def test_ideal_alternatives():
    path_to_task = TASKS_ROOT / '1_aircraft' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)

    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(task)
    _ = decision_maker.run()

    expected_joint_matrix_raw = [
        [2., 1500, 20000, 5.5, 5, 9],
        [2.5, 2700, 18000, 6.5, 3, 5],
        [1.8, 2000, 21000, 4.5, 7, 7],
        [2.2, 1800, 20000, 5.0, 5, 5],
    ]
    expected_joint_matrix = np.array(expected_joint_matrix_raw)
    assert np.array_equal(decision_maker._joint_decision_matrix.get_raw(), expected_joint_matrix), \
        'Joining for a single expert does not work'

    # in the original book example seems to contain errors during normalization. However,
    # in order to follow other computations we need to keep their normalized matrix
    their_normalized = [
        [.4671, .3662, .5056, .5063, .4811, .6708],
        [.5839, .6591, .4550, .5983, .2887, .3727],
        [.4204, .4882, .5308, .4143, .6736, .5217],
        [.5139, .4392, .5056, .4603, .4811, .3727],
    ]
    decision_maker._joint_decision_matrix._normalized = np.array(their_normalized)

    # this time joint matrix would not be calculated again
    res = decision_maker.run()

    expected_ideal_alternative = np.array([.1168, .0659, .0531, .0414, .1347, .2012])
    assert np.array_equal(decision_maker._artificial_ideal_alternative, expected_ideal_alternative), \
        'Ideal alternatives do not match'

    expected_ideal_negative_alternative = np.array([.0841, .0366, .0455, .0598, .0577, .1118])
    assert np.array_equal(decision_maker._artificial_ideal_negative_alternative, expected_ideal_negative_alternative), \
        'Ideal negative alternatives do not match'

    # and again it goes differently with a book example, there it is [.0545, .1197, .0580, .1009]
    expected_separation_measure_ideal = np.array([.0546, .1197, .0580, .1009])
    assert np.array_equal(decision_maker._separation_measure_from_ideal, expected_separation_measure_ideal), \
        'Separation measures to ideal solution do not match'

    expected_separation_measure_ideal_negative = np.array([.0983, .0439, .0920, .0458])
    assert np.array_equal(decision_maker._separation_measure_from_ideal_negative,
                          expected_separation_measure_ideal_negative), \
        'Separation measures to ideal negative solution do not match'

    expected_closeness = np.array([.6430, .2680, .6130, .3120])
    assert np.array_equal(decision_maker._relative_closeness, expected_closeness), \
        'Closeness measures to ideal solution do not match'

    expected_res_raw = [
        [0., .643],
        [2., .613],
        [3., .312],
        [1., .268],
    ]
    expected_res = np.array(expected_res_raw)
    assert np.array_equal(res, expected_res), 'Reported aggregation results do not match'


def test_ideal_alternatives_multi_expert_case():
    path_to_task = TASKS_ROOT / '2_aircraft_multiple_experts' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)

    raw_data_expert_1 = [
        [2., 1500, 20000, 5.5, 5, 9],
        [2.5, 2700, 18000, 6.5, 3, 5],
        [1.8, 2000, 21000, 4.5, 7, 7],
        [2.2, 1800, 20000, 5.0, 5, 5],
    ]
    expected_matrix_expert_1 = np.array(raw_data_expert_1)

    raw_data_expert_2 = [
        [3., 2250, 30000, 8.25, 9, 7],
        [3.75, 4050, 27000, 9.75, 5, 3],
        [2.7, 3000, 31500, 6.75, 9, 5],
        [3.3, 2700, 30000, 7.5, 7, 5],
    ]
    expected_matrix_expert_2 = np.array(raw_data_expert_2)

    assert len(task._decision_matrices) == 2, 'JSON contains 2 matrices'
    assert np.array_equal(task._decision_matrices['expert1'].get_raw(),
                          expected_matrix_expert_1), 'Load from JSON failed'
    assert np.array_equal(task._decision_matrices['expert2'].get_raw(),
                          expected_matrix_expert_2), 'Load from JSON failed'

    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(task)

    res = decision_maker.run()

    expected_joint_matrix_raw = [
        [2.4, 1800, 24000, 6.6, 6.6, 8.2],
        [3.0, 3240, 21600, 7.8, 3.8, 4.2],
        [2.16, 2400, 25200, 5.4, 7.8, 6.2],
        [2.64, 2160, 24000, 6.0, 5.8, 5],
    ]
    expected_joint_matrix = np.array(expected_joint_matrix_raw)
    assert np.array_equal(decision_maker._joint_decision_matrix.get_raw(), expected_joint_matrix), \
        'Joining for a single expert does not work'

    expected_res_raw = [
        [0., .716],
        [2., .579],
        [3., .337],
        [1., .269],
    ]
    expected_res = np.array(expected_res_raw)
    assert np.array_equal(res, expected_res), 'Reported aggregation results do not match'


def test_ideal_alternatives_multi_expert_case_no_expert_weights():
    path_to_task = TASKS_ROOT / '3_aircraft_multiple_experts_no_weights' / 'task.json'
    task = TaskModelFactory().from_json(path_to_task)

    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(task)

    res = decision_maker.run()

    expected_res_raw = [
        [0., .729],
        [2., .572],
        [3., .353],
        [1., .267],
    ]
    expected_res = np.array(expected_res_raw)
    assert np.array_equal(res, expected_res), 'Reported aggregation results do not match'
