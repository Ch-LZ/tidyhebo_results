"""Handle numerical divergence of Bayesmark's MLP-SGD regression estimator."""
import traceback
import warnings


def numerical_failure(error):
    message = str(error)
    known = (
        message == 'Input contains NaN.'
        or message.startswith("Input contains infinity or a value too large")
        or message.startswith('Solver produced non-finite parameter weights')
    )
    return known and any(
        frame.f_globals.get('__name__') in {
            'sklearn.neural_network._multilayer_perceptron',
            'sklearn.metrics._regression',
        }
        for frame, _ in traceback.walk_tb(error.__traceback__)
    )


def failure_penalty(target, metric):
    import numpy as np

    if metric == 'mae':
        scale = np.mean(np.abs(target - np.median(target)))
    elif metric == 'mse':
        scale = np.var(target)
    else:
        raise ValueError(f'Unexpected MLP regression metric: {metric}')
    return float(10 * max(1.0, scale))


def evaluate(test, config, penalty):
    """Keep CV and holdout independent, including when the holdout fit fails."""
    import numpy as np
    from bayesmark.sklearn_funcs import CV_SPLITS
    from sklearn.model_selection import cross_val_score

    params = dict(config)
    params.update(test.fixed_params)
    estimator = test.base_model(**params)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            cv = -float(np.mean(cross_val_score(
                estimator, test.data_X, test.data_y, scoring=test.scorer,
                cv=CV_SPLITS, error_score='raise')))
    except ValueError as error:
        if not numerical_failure(error):
            raise
        return penalty, None, 'cv_nonfinite'
    if not np.isfinite(cv):
        return penalty, None, 'cv_loss_nonfinite'

    estimator = test.base_model(**params)
    try:
        estimator.fit(test.data_X, test.data_y)
    except ValueError as error:
        if not numerical_failure(error):
            raise
        return cv, None, 'holdout_fit_nonfinite'
    try:
        held_out = -float(test.scorer(estimator, test.data_Xt, test.data_yt))
    except ValueError as error:
        if not numerical_failure(error):
            raise
        return cv, None, 'holdout_scoring_nonfinite'
    if not np.isfinite(held_out):
        return cv, None, 'holdout_loss_nonfinite'
    return cv, held_out, None
