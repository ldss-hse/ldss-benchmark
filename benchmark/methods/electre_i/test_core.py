import numpy as np

from benchmark.assessments.decision_matrix import DecisionMatrix, DecisionMatrixFactory
from benchmark.methods.electre_i.core import ElectreDecisionMaker


def test_ideal_alternatives():
    decision_matrix: DecisionMatrix = DecisionMatrixFactory.from_book_aircraft_example()
    decision_maker: ElectreDecisionMaker = ElectreDecisionMaker(decision_matrix)
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

    decision_maker.run()

    # all criteria IDs are subtracted with 1 as we are living in 0-indexing world
    expected_concordance = {
        (0, 1): {2, 3, 4, 5},
        (0, 2): {0, 5},
        (0, 3): {2, 4, 5},
        (1, 0): {0, 1},
        (1, 2): {0, 1},
        (1, 3): {0, 1, 5},
        (2, 0): {1, 2, 3, 4},
        (2, 1): {2, 3, 4, 5},
        (2, 3): {1, 2, 3, 4, 5},
        (3, 0): {0, 1, 2, 3, 4},
        (3, 1): {2, 3, 4, 5},
        (3, 2): {0},
    }
    assert np.array_equal(decision_maker._concordance, expected_concordance), 'Concordance sets do not match'

    expected_discordance = {
        (0, 1): {0, 1},
        (0, 2): {1, 2, 3, 4},
        (0, 3): {0, 1, 3},
        (1, 0): {2, 3, 4, 5},
        (1, 2): {2, 3, 4, 5},
        (1, 3): {2, 3, 4},
        (2, 0): {0, 5},
        (2, 1): {0, 1},
        (2, 3): {0},
        (3, 0): {5},
        (3, 1): {0, 1},
        (3, 2): {1, 2, 3, 4, 5},
    }
    assert np.array_equal(decision_maker._discordance, expected_discordance), 'Discordance sets do not match'

    expected_concordance_matrix_raw = [
        [0, .7, .5, .6],
        [.3, 0, .3, .6],
        [.5, .7, 0, .8],
        [.7, .7, .2, 0],
    ]
    expected_concordance_matrix = np.array(expected_concordance_matrix_raw)
    assert np.allclose(decision_maker._concordance_matrix, expected_concordance_matrix), \
        'Concordance matrix does not match'

    expected_discordance_matrix_raw = [
        [0, .3277, .8613, .1051],
        [1., 0, 1., 1.],
        [1., .4247, 0, .4183],
        [1., .5714, 1., 0]
    ]
    expected_discordance_matrix = np.array(expected_discordance_matrix_raw)
    assert np.allclose(decision_maker._discordance_matrix, expected_discordance_matrix), \
        'Discordance matrix does not match'
