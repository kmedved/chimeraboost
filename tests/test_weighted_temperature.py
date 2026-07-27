"""Weight-contract tests for classifier temperature calibration."""

import numpy as np
import pytest

from chimeraboost import ChimeraBoostClassifier
from chimeraboost.sklearn_api import _fit_temperature


def _duplicate_first_row(raw, y, weight):
    return (
        np.concatenate([raw[:1], raw], axis=0),
        np.concatenate([y[:1], y]),
        np.concatenate([[weight[0] / 2.0, weight[0] / 2.0], weight[1:]]),
    )


@pytest.mark.parametrize("multiclass", [False, True])
def test_weighted_temperature_ignores_zero_weight_rows(multiclass):
    if multiclass:
        raw = np.array(
            [
                [2.0, -0.5, -1.0],
                [-0.5, 1.5, -0.2],
                [-0.8, -0.4, 1.7],
                [0.8, 0.2, -0.3],
            ]
        )
        y = np.array([0, 1, 2, 0])
        extra_raw = np.array([[-20.0, 20.0, -20.0]])
        extra_y = np.array([0])
    else:
        raw = np.array([2.0, -1.5, 0.7, -0.4])
        y = np.array([1.0, 0.0, 1.0, 0.0])
        extra_raw = np.array([-20.0])
        extra_y = np.array([1.0])
    weight = np.array([0.5, 2.0, 1.5, 0.75])

    expected = _fit_temperature(
        raw, y, multiclass, sample_weight=weight
    )
    observed = _fit_temperature(
        np.concatenate([raw, extra_raw], axis=0),
        np.concatenate([y, extra_y]),
        multiclass,
        sample_weight=np.concatenate([weight, [0.0]]),
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("multiclass", [False, True])
def test_weighted_temperature_preserves_split_weight_duplicates(multiclass):
    if multiclass:
        raw = np.array(
            [
                [1.5, -0.2, -0.8],
                [-0.3, 1.1, -0.4],
                [-0.5, -0.1, 1.4],
                [0.7, 0.3, -0.6],
            ]
        )
        y = np.array([0, 1, 2, 0])
    else:
        raw = np.array([1.5, -1.1, 0.6, -0.3])
        y = np.array([1.0, 0.0, 1.0, 0.0])
    weight = np.array([2.0, 0.5, 1.25, 0.75])
    duplicate_raw, duplicate_y, duplicate_weight = _duplicate_first_row(
        raw, y, weight
    )

    expected = _fit_temperature(
        raw, y, multiclass, sample_weight=weight
    )
    observed = _fit_temperature(
        duplicate_raw,
        duplicate_y,
        multiclass,
        sample_weight=duplicate_weight,
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_threads_eval_weights_into_temperature(n_classes):
    rng = np.random.default_rng(20260727 + n_classes)
    X = rng.normal(size=(240, 5))
    coefficients = rng.normal(size=(5, n_classes))
    scores = np.sum(X[:, :, None] * coefficients[None, :, :], axis=1)
    if n_classes == 2:
        y = (scores[:, 1] > 0.0).astype(np.int64)
    else:
        y = np.argmax(scores, axis=1)
    X_eval = rng.normal(size=(80, 5))
    eval_scores = np.sum(
        X_eval[:, :, None] * coefficients[None, :, :], axis=1
    )
    if n_classes == 2:
        y_eval = (eval_scores[:, 1] > 0.0).astype(np.int64)
    else:
        y_eval = np.argmax(eval_scores, axis=1)
    eval_weight = np.linspace(0.25, 2.0, len(X_eval))

    extra_X = np.full((3, X.shape[1]), 100.0)
    extra_y = np.arange(3) % n_classes
    augmented_eval = (
        np.vstack([X_eval, extra_X]),
        np.concatenate([y_eval, extra_y]),
        np.concatenate([eval_weight, np.zeros(3)]),
    )
    base_eval = (X_eval, y_eval, eval_weight)
    params = {
        "n_estimators": 24,
        "early_stopping": False,
        "refit_full": False,
        "cross_features": False,
        "random_state": 17,
        "thread_count": 1,
    }

    base = ChimeraBoostClassifier(**params).fit(X, y, eval_set=base_eval)
    augmented = ChimeraBoostClassifier(**params).fit(
        X, y, eval_set=augmented_eval
    )

    np.testing.assert_array_equal(
        base.model_.predict_raw(X_eval),
        augmented.model_.predict_raw(X_eval),
    )
    np.testing.assert_allclose(
        augmented.temperature_, base.temperature_, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        augmented.predict_proba(X_eval),
        base.predict_proba(X_eval),
        rtol=0.0,
        atol=1e-12,
    )
