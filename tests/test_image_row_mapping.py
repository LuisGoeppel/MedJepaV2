import numpy as np
import pandas as pd
import pytest

from core.data import ensure_original_index, load_splits, EvalDataset
from core.config import load_run_config


def test_duplicate_ids_match_physical_rows_and_preserve_order():
    full = pd.DataFrame({
        "id": ["a", "a", "b", "b", "unique"],
        "patient": ["p", "p", "q", "q", "r"],
        "context": ['{"view":"CC"}', '{"view":"MLO"}', "same", "same", None],
        "original_birads": [1.0, 3.0, 2.0, 2.0, np.nan],
    }, index=[20, 40, 60, 80, 100])
    split = full.iloc[[1, 4, 0]].copy()
    split["collapsed_birads"] = "derived label is not an identity key"
    result = ensure_original_index(split, full, "train")
    assert result.original_index.tolist() == [1, 4, 0]
    assert result.index.tolist() == [40, 100, 20]
    pd.testing.assert_frame_equal(result.drop(columns="original_index"), split)
    # Duplicate IDs elsewhere in the full dataset should not prevent a unique match.
    assert ensure_original_index(full.iloc[[4]][["id"]], full, "val").original_index.tolist() == [4]
    with pytest.raises(ValueError, match="ambiguous or unmatched"):
        ensure_original_index(full.iloc[[2]], full, "train")
    conflicting = split.iloc[[0]].copy()
    conflicting["context"] = "not present"
    with pytest.raises(ValueError, match="ambiguous or unmatched"):
        ensure_original_index(conflicting, full, "train")
    with pytest.raises(ValueError, match="duplicate image rows"):
        ensure_original_index(pd.concat([split.iloc[[0]], split.iloc[[0]]]), full, "train")


def test_numeric_csv_inference_and_missing_metadata():
    full = pd.DataFrame({"id": ["a", "a", "b"], "patient": [1.0, 2.0, np.nan], "context": [None, None, None]})
    split = pd.DataFrame({"id": ["a"], "patient": [2], "context": [None]})
    assert ensure_original_index(split, full, "train").original_index.tolist() == [1]


def test_duplicate_id_splits_load_correct_mammogram(tiny_run):
    paths = load_run_config(tiny_run)["paths"]
    full = pd.read_csv(paths["full_csv"])
    full.loc[1, "id"] = full.loc[0, "id"]
    full.to_csv(paths["full_csv"], index=False)
    train = full.iloc[[1, 0, *range(2, 12)]].drop(columns="original_index")
    train.to_csv(paths["train_csv"], index=False)
    raw, mapped, _, _ = load_splits(*(paths[k] for k in ("full_csv", "train_csv", "val_csv", "test_csv")))
    assert mapped.original_index.tolist() == [1, 0, *range(2, 12)]
    images = np.memmap(paths["bin"], mode="r", dtype="uint16", shape=(len(raw), 32, 32))
    dataset = EvalDataset(mapped, paths["bin"], len(raw), (32, 32), "uint16", lambda image: image)
    for position, original_index in enumerate(mapped.original_index):
        image, _ = dataset[position]
        np.testing.assert_array_equal(image.numpy()[0, 0], images[original_index].astype(np.float32) / 65535)
