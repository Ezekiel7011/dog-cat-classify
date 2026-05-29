from aoi_inspection.metrics import compute_metrics


def test_compute_false_call_and_miss_rates():
    metrics = compute_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 0],
        classes=["pass", "defect"],
    )

    assert metrics.accuracy == 0.5
    assert metrics.false_call_rate_by_class["defect"] == 0.5
    assert metrics.miss_rate_by_class["defect"] == 0.5
