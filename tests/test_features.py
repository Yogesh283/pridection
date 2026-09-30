"""Feature engineering tests (no leakage)."""

from __future__ import annotations

from models.features import build_feature_vector, build_supervised_dataset


def _rounds(n: int):
    rows = []
    for i in range(n):
        number = i % 10
        color = "RED" if number % 2 == 0 else "GREEN"
        if number in (0, 5):
            color = f"{color},VIOLET"
        rows.append(
            {
                "period": str(2026000000 + i),
                "number": number,
                "color": color,
            }
        )
    return rows


def test_feature_generation_keys():
    feats = build_feature_vector(_rounds(60))
    assert "history_size" in feats
    assert "num_freq_w10_0" in feats
    assert "color_red_w20" in feats
    assert "number_streak" in feats
    assert "trans_num_3" in feats
    assert feats["history_size"] == 60


def test_features_empty_history():
    assert build_feature_vector([]) == {}


def test_supervised_dataset_no_future_leakage():
    rounds = _rounds(80)
    x, y_num, y_col, y_bs, keys = build_supervised_dataset(rounds, min_history=20)
    assert len(x) == len(y_num) == len(y_col) == len(y_bs)
    assert len(x) == 60
    # First label must equal round at index 20
    assert int(y_num[0]) == rounds[20]["number"]
    assert int(y_bs[0]) in (0, 1)
    assert keys
    assert "last_bs_big" in keys or "bs_mean_big_w3" in keys
