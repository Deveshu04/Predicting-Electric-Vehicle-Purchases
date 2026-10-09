import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["01-eda", "02-models", "03-stack"]
CONCLUSION = "**Conclusion.**"
PENDING = "Pending run A."
DASHES = (chr(0x2013), chr(0x2014))
PUBLIC_FILES = {
    "train.csv", "test.csv", "sample_submission.csv",
    "oof_six_views.csv", "test_six_views.csv", "oof_realmlp_g.csv", "test_realmlp_g.csv",
    "oof_hybrid_3seed.npy", "test_hybrid_3seed.npy",
    "oof_lgb.npy", "test_lgb.npy", "oof_cat.npy", "test_cat.npy", "oof_xgb.npy", "test_xgb.npy",
    "v19_oof.csv", "v19_test.csv", "yek_oof.csv", "yek_test.csv", "jaz_oof.csv", "jaz_test.csv",
    "GENERATOR_AWARE_LOGREG_SAMPLE_OOF.parquet", "GENERATOR_AWARE_LOGREG_SAMPLE_TEST.parquet",
    "LOGREG_SAMPLE_OOF.parquet", "LOGREG_SAMPLE_TEST.parquet",
    "XGB_SAMPLE_OOF.parquet", "XGB_SAMPLE_TEST.parquet",
}


def notebook(name):
    return json.loads((ROOT / "notebooks" / name / f"{name}.ipynb").read_text(encoding="utf-8"))


def metadata(name):
    return json.loads((ROOT / "notebooks" / name / "kernel-metadata.json").read_text(encoding="utf-8"))


def cells(name):
    return [(cell["cell_type"], "".join(cell["source"])) for cell in notebook(name)["cells"]]


def code(name):
    return "\n".join(source for kind, source in cells(name) if kind == "code")


@pytest.mark.parametrize("name", NAMES)
def test_opens_with_title_and_context(name):
    kind, source = cells(name)[0]
    lines = source.splitlines()
    assert kind == "markdown"
    assert lines[0].startswith("# ")
    assert any(len(line.split()) >= 20 for line in lines[1:])


@pytest.mark.parametrize("name", NAMES)
def test_every_code_cell_is_framed(name):
    kinds = cells(name)
    for index, (kind, source) in enumerate(kinds):
        if kind != "code":
            continue
        before_kind, before = kinds[index - 1]
        assert index > 0 and before_kind == "markdown" and not before.startswith(CONCLUSION), index
        assert index + 1 < len(kinds), index
        after_kind, after = kinds[index + 1]
        assert after_kind == "markdown" and after.startswith(CONCLUSION), index


@pytest.mark.parametrize("name", NAMES)
def test_smoke_switch_once(name):
    assert code(name).count("SMOKE = False") == 1
    assert "SMOKE = True" not in code(name)


@pytest.mark.parametrize("name", NAMES)
def test_code_parses_without_comments(name):
    for kind, source in cells(name):
        if kind == "code":
            ast.parse(source)
            assert not any(line.lstrip().startswith("#") for line in source.splitlines())


@pytest.mark.parametrize("name", NAMES)
def test_text_is_clean(name):
    for kind, source in cells(name):
        assert source.isascii()
        assert not any(dash in source for dash in DASHES)
        for word in ("TODO", "TBD", "FIXME", "conclude(", "report("):
            assert word not in source


@pytest.mark.parametrize("name", NAMES)
def test_conclusions_written(name):
    for kind, source in cells(name):
        assert PENDING not in source


@pytest.mark.parametrize("name", NAMES)
def test_kernel_metadata(name):
    meta = metadata(name)
    assert meta["code_file"] == f"{name}.ipynb"
    assert meta["id"] == f"deveshupathak/ev-purchase-{name}"
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is False
    assert meta["competition_sources"] == ["playground-series-s6e9"]
    assert notebook(name)["metadata"]["kernelspec"]["name"] == "python3"


def test_stack_reads_models_notebook():
    assert "deveshupathak/ev-purchase-02-models" in metadata("03-stack")["kernel_sources"]


def test_saved_names_do_not_clash_with_lookups():
    models = code("02-models")
    saved = set(re.findall(r'np\.savez\("([^"]+)"', models)) | set(re.findall(r'Path\("([^"]+)"\)\.write_text', models))
    assert saved == {"own_predictions.npz", "lgb_params.json"}
    assert not saved & PUBLIC_FILES
    assert '"/kaggle/input/**/own_predictions.npz"' in code("03-stack")


def test_stack_checks_prediction_ids():
    stack = code("03-stack")
    assert "assert len(own_file) == 1" in stack
    assert 'np.array_equal(saved["train_id"], train["id"].to_numpy())' in stack
    assert 'np.array_equal(saved["test_id"], test["id"].to_numpy())' in stack
