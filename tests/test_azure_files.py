"""Checks on the Azure files that run without Azure: job specs, environment, workflows.

They catch mistakes that would otherwise only show up when a job fails in the
cloud: a typo in a script name, packages that differ from requirements.txt,
or a workflow that uses a stored secret instead of OIDC.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
JOBS = sorted((ROOT / "jobs").glob("*.yml"))
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
DEV_ONLY = {"pytest", "ruff"}  # in requirements.txt, not needed in Azure ML


def pins(lines: list[str]) -> dict[str, str]:
    """{'pandas': '3.1.1', ...} from lines like 'pandas==3.1.1  # comment'."""
    found = {}
    for line in lines:
        match = re.match(r"\s*-?\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def conda_pins() -> dict[str, str]:
    conda = yaml.safe_load((ROOT / "environment" / "conda.yml").read_text())
    pip_list = next(d["pip"] for d in conda["dependencies"] if isinstance(d, dict))
    return pins(pip_list)


JOB_NAMES = ["train-register.yml", "batch-score.yml"]


def test_there_are_two_jobs():
    assert [job.name for job in JOBS] == ["batch-score.yml", "train-register.yml"]


@pytest.mark.parametrize("job_file", JOBS, ids=lambda p: p.name)
def test_job_spec_is_complete(job_file: Path):
    job = yaml.safe_load(job_file.read_text())
    assert job["type"] == "command"
    assert job["compute"].startswith("azureml:")
    assert (job_file.parent / job["code"] / "src").is_dir()
    assert (job_file.parent / job["environment"]["conda_file"]).is_file()
    # Every script the command runs exists.
    for module in re.findall(r"python -m src\.(\w+)", job["command"]):
        assert (ROOT / "src" / f"{module}.py").is_file(), f"src/{module}.py not found"
    # Every ${{inputs.x}} / ${{outputs.x}} in the command is defined.
    for kind, name in re.findall(r"\$\{\{(inputs|outputs)\.(\w+)\}\}", job["command"]):
        assert name in job.get(kind, {}), f"{kind}.{name} is used but not defined"


def test_conda_has_the_same_versions_as_requirements():
    required = pins((ROOT / "requirements.txt").read_text().splitlines())
    in_azure = conda_pins()
    for package, version in required.items():
        if package not in DEV_ONLY:
            assert in_azure.get(package) == version, f"{package}: {in_azure.get(package)}"


def test_laptop_and_jobs_use_the_same_azureml_mlflow():
    azure = pins((ROOT / "requirements-azure.txt").read_text().splitlines())
    assert azure["azureml-mlflow"] == conda_pins()["azureml-mlflow"]
    deploy = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    assert "pip install -r requirements-azure.txt" in deploy


@pytest.mark.parametrize("workflow_file", WORKFLOWS, ids=lambda p: p.name)
def test_workflows_use_oidc_and_no_secrets(workflow_file: Path):
    text = workflow_file.read_text()
    uses_secret = re.search(r"secrets\.\w", text)
    assert not uses_secret, "use OIDC and GitHub variables, never stored secrets"
    if "azure/login" in text:
        workflow = yaml.safe_load(text)
        assert workflow["permissions"]["id-token"] == "write"
        for name in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_SUBSCRIPTION_ID"]:
            assert f"vars.{name}" in text


def test_fabric_notebook_is_valid():
    notebook = json.loads((ROOT / "notebooks" / "fabric_batch_score.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


def test_both_jobs_run_on_the_projects_own_cluster():
    """One cluster per project: training and scoring use the same one."""
    jobs = [yaml.safe_load((ROOT / "jobs" / name).read_text()) for name in JOB_NAMES]
    computes = {job["compute"] for job in jobs}
    assert len(computes) == 1


def test_mlflow_stays_on_a_version_azure_ml_supports():
    """Azure ML supports MLflow up to 2.16; MLflow 3 fails there when saving a model."""
    major, minor, *_ = pins((ROOT / "requirements.txt").read_text().splitlines())["mlflow"].split(
        "."
    )
    assert (int(major), int(minor)) <= (2, 16)
