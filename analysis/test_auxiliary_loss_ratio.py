import numpy as np
import pandas as pd

from analysis.auxiliary_loss_ratio import ratio_frame


def test_ratio_of_means_and_latest_duplicate():
    frame = pd.DataFrame(dict(optimizer_step=[2, 1, 2, 3],
        main_elbo=[100., 1., 3., 2.], np_prev=[100., 2., 2., 2.],
        np_next=[100., 2., 2., 2.]))
    result = ratio_frame(frame, {-1: .25, 1: .25}, smooth=2)
    np.testing.assert_allclose(result.aux_to_main, [1., .5, .4])
    np.testing.assert_allclose(result.aux_share_total, [.5, 1/3, 2/7])
    assert result.optimizer_step.tolist() == [1, 2, 3]


def test_gaps_do_not_extend_optimizer_window_and_invalid_rows_excluded():
    frame = pd.DataFrame(dict(optimizer_step=[1, 2, 100, 101],
        main_elbo=[1., 2., 4., np.nan], np_prev=[2., 2., 2., np.nan],
        np_next=[2., 2., 2., np.nan]))
    result = ratio_frame(frame, {-1: .25, 1: .25}, smooth=2)
    assert result.optimizer_step.tolist() == [1, 2, 100]
    np.testing.assert_allclose(result.aux_to_main, [1., 2/3, .25])
