"""Tests of the tutorial, `TUTORIAL.md`: its commands, and what it says that they print.

The outputs that the tutorial shows are the ones of two real runs of DrBERT. The last test types
its commands in a copy of the repository, with a tiny model on the synthetic dataset of the task,
and compares what they print with what the tutorial shows.
"""

import json
import re
import shlex
import shutil
import sys

import pytest

from helpers import RECIPES, REPO, TASKS, Task, copy_tools, load_module, read_json, run_command

TUTORIAL = REPO / "TUTORIAL.md"
TASK = "quaero-ner_medline"
MODEL = "Dr-BERT/DrBERT-7GB"
# The first command of the tutorial: the next ones do the same runs
TRAINING = ["python", "run.py", "--tasks", TASK, "--models", MODEL, "--offline", "false", "--epochs", "5"]
# The runs of the tutorial, by seed: their names are built from the arguments of this command
RUNS = ["DrBenchmark-quaero-ner_medline-3af4cf7540034bff", "DrBenchmark-quaero-ner_medline-e5de16c4954b4c11"]
# Where the tutorial says that the repository is, in the outputs that have its path
REPOSITORY = "/path/to/DrBenchmark"
# What is typed for each output that the tutorial shows
SCRIPTS = [
    "run.py", "run.py", "run.py", "tools/CollectAndAverageScores.py", "tools/BuildExcelTableOfScores.py",
    "predict.py", "predict.py", "tools/ShowPredictionErrors.py", "tools/RemoveKeptModels.py",
]


def tutorial():
    return TUTORIAL.read_text(encoding="utf-8")


def blocks():
    """The blocks of the tutorial, in their order: (language, text)."""
    return re.findall(r"^```(\w+)\n(.*?)^```$", tutorial(), re.M | re.S)


def steps():
    """The commands that the tutorial shows the output of: [(arguments of the command, lines of the output)]."""
    all_blocks = blocks()
    return [
        (shlex.split(text), output.splitlines())
        for (language, text), (next_language, output) in zip(all_blocks, all_blocks[1:])
        if (language, next_language) == ("bash", "plain")
    ]


def shown(lines):
    """The lines of an output that the tutorial shows whole: `...` is what it cuts."""
    return [line for line in lines if line.strip() and "..." not in line]


def without_date(line):
    """A line without the time that `run.py` writes at the beginning of its messages."""
    return re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ", "", line)


def shape(line):
    """A line without what depends on the model that is trained: the names of its runs, its numbers and the width of the columns."""
    line = re.sub(r"DrBenchmark-quaero-ner_medline-[0-9a-f]{16}", "RUN", line)
    line = re.sub(r"\d+(\.\d+)?", "N", line)
    return re.sub(r"\s+", " ", re.sub(r"N (bytes|kB|MB|GB)", "SIZE", line)).strip()


def words(labelled):
    """The words of a sentence that `predict.py` printed with its entities: `Une [toux sèche]/DISO`."""
    return re.sub(r"\]?/\S+", "", labelled).replace("[", "").split()


def anchors(path):
    """The anchors of the titles of a Markdown file, as GitHub builds them."""
    titles = re.findall(r"^#+ (.*)$", path.read_text(encoding="utf-8"), re.M)
    return {re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-") for title in titles}


def test_tutorial_is_linked_and_its_links_lead_somewhere():
    assert "[TUTORIAL.md](./TUTORIAL.md)" in (REPO / "README.md").read_text(encoding="utf-8")
    links = [link for link in re.findall(r"\]\(([^)]*)\)", tutorial()) if not link.startswith("https://")]
    assert len(links) > 15
    for link in links:
        path, _, anchor = link.partition("#")
        assert path == "" or path.startswith("./"), link
        assert (REPO / path).exists(), link
        if anchor:
            assert anchor in anchors(REPO / path if path else TUTORIAL), link


def test_tutorial_shows_the_output_of_its_commands():
    """Each command is followed by what it prints, but the installation and the removal of the runs."""
    languages = [language for language, _ in blocks()]
    assert languages[0] == languages[-1] == "bash" and languages.count("json") == 1
    assert steps()[0][0] == TRAINING
    assert [command[:2] for command, _ in steps()] == [["python", script] for script in SCRIPTS]
    assert all((REPO / script).is_file() for script in SCRIPTS)
    # The second command is the first one, and the third one adds a run
    assert [command[len(TRAINING):] for command, _ in steps()[:3]] == [[], [], ["--nb-run", "2"]]
    assert all(command[:len(TRAINING)] == TRAINING for command, _ in steps()[:3])
    # The installation is the one of the README
    assert blocks()[0][1] in (REPO / "README.md").read_text(encoding="utf-8")


def test_tutorial_says_what_the_repository_has():
    """The numbers that the tutorial gives: the tasks, the epochs, the runs and the size of the training set."""
    text = tutorial()
    assert len(TASKS) == 27 and "one of the 27 names" in text
    assert Task(TASK).config["epochs"] == 15 and "`--epochs 5`: 5 epochs instead of the 15 of the task" in text
    assert load_module("number_of_runs", REPO / "tools" / "number_of_runs.py").NB_RUNS == 4
    assert "4 runs in the paper" in text and "The table expects 4 runs for a task" in text and "`python run.py --tasks all --nb-run 4`" in text
    # The table of the subsets, in the README of the recipe: training, validation and test sets
    sizes = re.search(r"^\| `medline` \|\s*(\d+) \|\s*\d+ \|\s*(\d+) \|$", (RECIPES / "quaero" / "README.md").read_text(encoding="utf-8"), re.M)
    assert f"with {sizes[1]} titles to train on" in text and f"{sizes[2]} examples of the test set" in text


def test_runs_of_the_tutorial_have_the_names_it_shows(utils, monkeypatch):
    """The name of a run is built from its arguments: the ones of the tutorial name the runs that it shows.

    `max_position_embeddings` is given with the value of `config.yaml` that the tutorial was typed
    with, for the test not to depend on that file.
    """
    task = Task(TASK)
    monkeypatch.chdir(task.scripts_dir)
    # What `run.py` starts for the task, without the seed: with one, a run that exists stops the script
    monkeypatch.setattr(sys, "argv", [*task.command(MODEL)[1:], *TRAINING[6:], "--max_position_embeddings", "512"])
    arguments = vars(utils.parse_args())
    assert [utils.create_run_name({**arguments, "seed": seed}) for seed in (0, 1)] == RUNS

    text = tutorial()
    assert set(re.findall(r"DrBenchmark-quaero-ner_medline-[0-9a-f]{16}", text)) == set(RUNS)
    # The results and the model of the first run, then what the tools say of each run
    assert f"`recipes/quaero/runs/{RUNS[0]}.json`" in text and f"`recipes/quaero/save_models/{RUNS[0]}_best_model/`" in text
    assert f"Run: {REPOSITORY}/recipes/quaero/runs/{RUNS[0]}.json\nTask {TASK}, model {MODEL}, seed 0 (other seeds: 1)\n" in text
    assert f"{REPOSITORY}/recipes/quaero/save_models/{RUNS[1]}_best_model  445.3 MB  ({TASK}, {MODEL}, seed 1)\n" in text
    # The removal of the runs: the models that the tool finds from the results, then the results
    removal = blocks()[-1][1].splitlines()
    assert removal == [
        f"python tools/RemoveKeptModels.py --task {TASK} --model {MODEL} --all --remove",
        "rm " + " ".join(f"recipes/quaero/runs/{run}.json" for run in RUNS),
    ]


@pytest.mark.e2e
def test_commands_of_the_tutorial_print_what_it_shows(tiny_bert, synthetic_data, tmp_path):
    """The tutorial, typed in a copy of the repository with a tiny model: each line that it shows is printed.

    The numbers and the names of the runs are not the ones of DrBERT: the lines are compared
    without them. The lines that depend on what a model predicts, the confusions and the examples
    of `ShowPredictionErrors.py`, are not compared.
    """
    repository = tmp_path / "DrBenchmark"
    repository.mkdir()
    copy_tools(repository)
    shutil.copy(REPO / "config.yaml", repository)
    # Neither the runs nor the data of this repository
    not_copied = shutil.ignore_patterns("__pycache__", "logs", "runs", "save_models", "data")
    shutil.copytree(REPO / "metrics", repository / "metrics", ignore=not_copied)
    for name in ("common", "quaero"):
        shutil.copytree(RECIPES / name, repository / "recipes" / name, ignore=not_copied)
    recipe = repository / "recipes" / "quaero"
    (recipe / "runs").mkdir()
    # The data, where the task reads them in offline mode
    shutil.copytree(synthetic_data / "quaero" / "local_hf_medline", recipe / "data" / "local_hf_medline")

    def results():
        return sorted(recipe.glob("runs/*.json"), key=lambda path: read_json(path)["hyperparameters"]["seed"])

    def kept_models():
        return sorted(path.name for path in recipe.glob("save_models/*_best_model"))

    def type_command(command):
        """Type a command of the tutorial for the tiny model, on the local data and for one epoch: the lines that it prints."""
        arguments = [str(tiny_bert) if argument == MODEL else argument for argument in command[1:]]
        if arguments[0] == "run.py":
            assert arguments[5:9] == ["--offline", "false", "--epochs", "5"]
            arguments[5:9] = ["--offline", "true", "--epochs", "1"]
        if arguments[0] == "tools/ShowPredictionErrors.py":
            # The test set is the local one too
            arguments.append("--offline")
        completed = run_command([sys.executable, *arguments], cwd=repository)
        assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]
        # As the tutorial writes them, with its model and its repository. A model that is given by its path is named
        # by it, but by `predict.py` when it says which model predicts: by the name of its directory
        printed = (completed.stdout + completed.stderr).replace(str(tiny_bert), MODEL).replace(f"Model: {tiny_bert.name} ", f"Model: {MODEL} ")
        printed = printed.replace(str(repository), REPOSITORY)
        return [without_date(line) for line in printed.splitlines()]

    for number, (command, output) in enumerate(steps()):
        printed = type_command(command)
        expected = [without_date(line) for line in shown(output)]
        assert expected, command
        if command[1] in ("run.py", "tools/CollectAndAverageScores.py"):
            # Messages that do not depend on the model: the runs that are computed and the ones that are not
            assert [line for line in expected if line not in printed] == [], command
        elif command[1] == "predict.py" and "--list" not in command:
            # The model that is taken, then the sentence with its entities: the words are the same, not the entities
            model, sentence = expected
            assert shape(model) in map(shape, printed)
            assert words(sentence) == "Une femme de 44 ans était hospitalisée pour une toux sèche .".split()
            assert words(sentence) in map(words, printed)
        else:
            # The lines that do not depend on what the model predicts: the other ones are indented, or start an example
            expected = [line for line in expected if not line.startswith((" ", "["))]
            assert len(expected) >= 2 and [line for line in expected if shape(line) not in map(shape, printed)] == [], command
        # Each run that is computed writes its results and keeps its model: the first command computes one, the third one another
        assert sorted(path.stem + "_best_model" for path in results()) == kept_models()
        assert len(kept_models()) == (1 if number < 2 else 2), command

    # The scores that are collected: the two runs of the task, as the tutorial shows them
    collected = read_json(repository / "stats" / "results.json")
    of_the_tutorial, = [json.loads(text) for language, text in blocks() if language == "json"]
    assert list(of_the_tutorial) == [MODEL] and list(collected) == [str(tiny_bert)]
    assert {task: {score: len(runs) for score, runs in scores.items()} for task, scores in collected[str(tiny_bert)].items()} \
        == {task: {score: len(runs) for score, runs in scores.items()} for task, scores in of_the_tutorial[MODEL].items()} \
        == {"quaero|ner_medline|1.0": {"overall_f1": 2, "overall_accuracy": 2}}

    # 6. `--remove` removes the model that `predict.py` does not take: the results stay
    printed = type_command(["python", "tools/RemoveKeptModels.py", "--remove"])
    assert len(kept_models()) == 1 and len(results()) == 2, printed

    # What next: the two runs are removed. Their names are not the ones of the runs of DrBERT
    first, second = blocks()[-1][1].splitlines()
    type_command(shlex.split(first))
    assert kept_models() == [] and len(results()) == 2
    for shown_name, path in zip(RUNS, results()):
        second = second.replace(shown_name, path.stem)
    assert run_command(shlex.split(second), cwd=repository).returncode == 0
    assert results() == [] and "No run in " in " ".join(type_command(["python", "predict.py", "--list"]))
