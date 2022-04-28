import numpy as np

from benchmark.assessments.decision_matrix import DecisionMatrix, DecisionMatrixFactory
from benchmark.methods.topsis.core import TopsisDecisionMaker


def test_ideal_alternatives():
    decision_matrix: DecisionMatrix = DecisionMatrixFactory.from_book_aircraft_example()
    decision_maker: TopsisDecisionMaker = TopsisDecisionMaker(decision_matrix)
    criteria_weights = (.2, .1, .1, .1, .2, .3)
    alternatives_type = (True, True, True, False, True, True)
    decision_maker.set_criteria_weights(criteria_weights)
    decision_maker.set_alternatives_type(alternatives_type)

    # in the original book example seems to contain errors during normalization. However,
    # in order to follow other computations we need to keep their normalized matrix
    their_normalized = [
        [.4671, .3662, .5056, .5063, .4811, .6708],
        [.5839, .6591, .4550, .5983, .2887, .3727],
        [.4204, .4882, .5308, .4143, .6736, .5217],
        [.5139, .4392, .5056, .4603, .4811, .3727],
    ]
    decision_matrix._normalized = np.array(their_normalized)

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
