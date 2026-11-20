"""Tests of the guide that adds a task, `recipes/ADDING_A_TASK.md`, and of the recipe it starts from.

`recipes/template/` is not a recipe of the benchmark: no task of `run.py` uses it. The last
test follows the guide in a copy of the repository, and runs the task that it adds.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys

import pytest
import yaml

import synthetic
from helpers import ALL_TASKS, RECIPE_NAMES, RECIPES, REPO, TASKS, TEMPLATE, Task, copy_tools, read_json, run_command, run_tool

GUIDE = RECIPES / "ADDING_A_TASK.md"
# The task that the guide adds: its line of `run.py`
LINE_OF_RUN_PY = "    'mycorpus-cls': Script('finetuning_bert_cls.py', 'cls.yaml'),\n"


def code_blocks(language):
    """The blocks of code of the guide that are in a language."""
    return re.findall(rf"^```{language}\n(.*?)^```$", GUIDE.read_text(encoding="utf-8"), re.M | re.S)


def test_template_is_not_a_recipe_of_the_benchmark():
    assert TEMPLATE.is_dir() and "template" not in RECIPE_NAMES
    assert not any(name.startswith(("template-", "mycorpus-")) for name in TASKS)


def test_template_has_the_files_of_a_recipe():
    files = sorted(str(path.relative_to(TEMPLATE)) for path in TEMPLATE.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    written = ["README.md", "scripts/finetuning_bert_cls.py", "scripts/run.sh", "yaml/cls.yaml"]
    # The directories that the runs read and write in: git keeps them for the file that makes it ignore their content
    kept = [f"{directory}/.gitignore" for directory in ("data", "runs", "save_models", "scripts/logs")]
    assert files == sorted(written + kept)
    reference = RECIPES / "cas"
    assert all((TEMPLATE / name).read_bytes() == (reference / name).read_bytes() for name in kept)
    # The guide lists them
    guide = GUIDE.read_text(encoding="utf-8")
    assert all(f"| `{name}`" in guide for name in written) and "| `data/`, `runs/`, `save_models/`" in guide


def test_script_of_the_template_starts_as_the_other_ones():
    """What a script does before it states its task: it finds the shared code and reads the arguments of its run."""
    def beginning(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[lines.index("import sys"):lines.index("from common import sequence_classification  # noqa: E402") + 1]

    script = TEMPLATE / "scripts" / "finetuning_bert_cls.py"
    reference = Task("cas-cls")
    assert beginning(script) == beginning(reference.scripts_dir / reference.script)

    lines = script.read_text(encoding="utf-8").splitlines()
    assert sum(line.startswith("TASK = ") for line in lines) == 1
    assert lines[-2:] == ["if __name__ == '__main__':", "    TASK.train()"]


def test_yaml_file_of_the_template_has_the_keys_of_the_other_ones():
    config = yaml.safe_load((TEMPLATE / "yaml" / "cls.yaml").read_text(encoding="utf-8"))
    reference = Task("cas-cls").config
    assert set(config) == set(reference)
    assert (config["task"], config["dataset"], config["subset"], config["fewshot"]) == ("mycorpus-cls", "DrBenchmark/MyCorpus", None, 1.0)
    assert [config[name] for name in ("output_dir", "run_dir", "data_dir")] == [reference[name] for name in ("output_dir", "run_dir", "data_dir")]


def test_launcher_of_the_template_is_the_one_of_a_recipe():
    launcher = TEMPLATE / "scripts" / "run.sh"
    assert os.access(launcher, os.X_OK)
    reference = Task("diamed-cls")
    expected = reference.launcher_path.read_text().replace(reference.script, "finetuning_bert_cls.py")
    assert launcher.read_text() == expected


def test_readme_of_the_template_documents_its_task():
    """As the README of a recipe has to: the task, its launcher and its YAML file."""
    readme = (TEMPLATE / "README.md").read_text(encoding="utf-8")
    assert all(text in readme for text in ("mycorpus-cls", "`run.sh`", "`yaml/cls.yaml`"))


def test_guide_shows_the_template():
    """The task of the template, the line of `run.py` and the synthetic dataset: the guide shows what the last test uses."""
    script = (TEMPLATE / "scripts" / "finetuning_bert_cls.py").read_text(encoding="utf-8").splitlines()
    task, line_of_run_py, dataset = code_blocks("python")
    # The lines of the script that state its task, without its comments
    shown = [line for line in task.splitlines() if line]
    assert shown == [line for line in script[script.index("def build_text(e, tokenizer):"):-3] if line and not line.lstrip().startswith("#")]
    assert line_of_run_py == LINE_OF_RUN_PY
    assert dataset.startswith('        "mycorpus": {\n')


def test_guide_names_what_exists():
    """The files that the guide links to, and what it names in them."""
    guide = GUIDE.read_text(encoding="utf-8")
    links = re.findall(r"\]\((\.[^)]*)\)", guide)
    assert len(links) > 10 and all((GUIDE.parent / link).exists() for link in links), links

    named = {
        "run.py": ["task2script = {", "Script("],
        "recipes/common/utils.py": ["subset2task_detail = {"],
        "tools/CollectAndAverageScores.py": ["def scores_of("],
        "download_datasets_locally.py": ["ds = ["],
        "tests/synthetic.py": ["def build_all("],
        "tools/BuildExcelTableOfScores.py": ["order = [", "dataset2pretty = {"],
    }
    for path, names in named.items():
        source = (REPO / path).read_text(encoding="utf-8")
        for name in names:
            assert name in source, (path, name)
            assert re.sub(r"\W.*", "", name.removeprefix("def ")) in guide, name
    # The kinds of tasks, and the options of their scripts
    for module in ("token_classification", "sequence_classification", "multi_label_classification", "regression"):
        assert f"`{module}`" in guide and (RECIPES / "common" / f"{module}.py").is_file()
    # The names of tasks that the tools know
    kinds = set(re.findall(r'"(\w+)" in task\b', (REPO / "tools" / "CollectAndAverageScores.py").read_text()))
    assert kinds == {"ner", "pos", "cls", "mcqa", "regr"} == {task.kind for task in ALL_TASKS}
    assert all(f"`{kind}`" in guide for kind in kinds)


@pytest.mark.e2e
def test_guide_adds_a_task_that_runs(tiny_bert, tmp_path):
    """The steps of the guide, in a copy of the repository: the task is trained, its scores are collected and its model predicts."""
    repository = tmp_path / "repository"
    repository.mkdir()
    copy_tools(repository)
    shutil.copy(REPO / "config.yaml", repository)
    not_copied = shutil.ignore_patterns("__pycache__", "logs")
    shutil.copytree(REPO / "metrics", repository / "metrics", ignore=not_copied)
    for name in ("common", "template"):
        shutil.copytree(RECIPES / name, repository / "recipes" / name, ignore=not_copied)

    # The recipe is a copy of the template
    copy, = [line for block in code_blocks("bash") for line in block.splitlines() if line.startswith("cp ")]
    assert subprocess.run(copy.split(), cwd=repository).returncode == 0
    recipe = repository / "recipes" / "mycorpus"

    # 3. The line of `run.py`
    run_py = (repository / "run.py").read_text(encoding="utf-8")
    assert run_py.count("task2script = {\n") == 1
    (repository / "run.py").write_text(run_py.replace("task2script = {\n", "task2script = {\n" + LINE_OF_RUN_PY), encoding="utf-8")

    # 4. The data, where the recipe reads them in offline mode: the synthetic dataset of the guide
    datasets = eval("{" + code_blocks("python")[2] + "}", vars(synthetic))
    datasets["mycorpus"][None].save_to_disk(str(recipe / "data" / "local_hf_None"))

    # 5. The three commands, with a tiny model
    train, collect, predict = [shlex.split(line) for line in code_blocks("bash")[1].splitlines()]
    assert (train[:2], collect[:2], predict[:2]) == (["python", "run.py"], ["python", "tools/CollectAndAverageScores.py"], ["python", "predict.py"])
    with_the_tiny_model = [str(tiny_bert) if argument == "Dr-BERT/DrBERT-7GB" else argument for argument in train[2:]]
    completed = run_command([sys.executable, repository / "run.py", *with_the_tiny_model], cwd=repository)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]
    assert "INFO::1 run(s): 1 computed, 0 already computed, 0 failed" in completed.stderr
    (results,) = (recipe / "runs").glob("DrBenchmark-mycorpus-cls-*.json")

    completed = run_tool("CollectAndAverageScores.py", repository, cwd=repository)
    assert completed.returncode == 0, completed.stderr[-3000:]
    collected = read_json(repository / "stats" / "results.json")
    assert {model: list(tasks) for model, tasks in collected.items()} == {str(tiny_bert): ["mycorpus|cls|1.0"]}
    assert set(collected[str(tiny_bert)]["mycorpus|cls|1.0"]) == {"macro_f1", "weighted_f1"}

    # A sentence that the tiny model knows the words of
    assert predict[2:6] == ["--task", "mycorpus-cls", "--model", "Dr-BERT/DrBERT-7GB"] and len(predict) == 7
    completed = subprocess.run(
        [sys.executable, str(repository / "predict.py"), "--task", "mycorpus-cls", "--model", str(tiny_bert), "le patient présente une douleur"],
        cwd=repository, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() in ("first", "second", "third")

    # The launcher of the recipe does the same run: it is already computed
    completed = run_command(f"bash recipes/mycorpus/scripts/run.sh {tiny_bert} --seed 0 --epochs 1", cwd=repository)
    assert completed.returncode == 0 and "This configuration was already computed" in completed.stdout
    assert [path.name for path in (recipe / "runs").glob("*.json")] == [results.name]
