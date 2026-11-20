"""Tests of `recipes/common/utils.py`: command line parsing and naming of the runs."""

import os
import re
import subprocess
import sys

import pytest
import yaml

from helpers import ALL_TASKS, RECIPES, REPO, Task, relative_path

MODEL = "Dr-BERT/DrBERT-7GB"
GLOBAL_CONFIG = yaml.safe_load((REPO / "config.yaml").read_text())


def task_ids(task):
    return task.name


@pytest.fixture
def parse(utils, monkeypatch):
    """Parse the arguments as the script of a task does.

    The script is run with the arguments given by its launcher, followed by `extra_args`.
    It is started from `cwd`: from its `scripts/` directory by default.
    """
    def parse(task, *extra_args, model=MODEL, cwd=None):
        if isinstance(task, str):
            task = Task(task)
        script = task.script if cwd is None else relative_path(task.scripts_dir / task.script, cwd)
        monkeypatch.chdir(task.scripts_dir if cwd is None else cwd)
        monkeypatch.setattr(sys, "argv", [script, *task.script_args(model, cwd), *extra_args])
        return utils.parse_args()

    return parse


@pytest.fixture
def starting_directories(tmp_path):
    """Directories a script can be started from, other than its `scripts/` directory."""
    def directories(task):
        other_recipe = "essai" if task.recipe != "essai" else "cas"
        return {
            "the root of the repository": REPO,
            "a directory out of the repository": tmp_path,
            # Its `../yaml/` directory has files with the same names as the one of the task
            "the scripts of another recipe": RECIPES / other_recipe / "scripts",
        }

    return directories


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_task_name_and_subset(task, parse):
    """The name used in `run.py` is the one written in the results of the run."""
    args = parse(task)
    assert args.task == task.name
    assert args.subset == task.subset
    assert args.dataset == task.config["dataset"]


def test_yaml_file_gives_the_default_values(parse):
    task = Task("cas-pos")
    args = parse(task)
    for name in ("epochs", "batch_size", "weight_decay", "learning_rate", "fewshot", "data_dir", "run_dir"):
        assert getattr(args, name) == task.config[name], name
    assert args.output_dir == task.config["output_dir"].rstrip("/")


def test_global_config_is_applied(parse):
    args = parse("cas-pos")
    for name, value in GLOBAL_CONFIG.items():
        assert getattr(args, name) == value, name


def test_command_line_overrides_the_yaml_file(parse):
    args = parse(
        "cas-pos",
        "--epochs", "3", "--batch_size", "2", "--learning_rate", "0.5", "--weight_decay", "0.25",
        "--seed", "7", "--fewshot", "0.5", "--max_position_embeddings", "128", "--run_dir", "/tmp/somewhere/",
    )
    assert (args.epochs, args.batch_size, args.learning_rate, args.weight_decay) == (3, 2, 0.5, 0.25)
    assert (args.seed, args.fewshot, args.max_position_embeddings) == (7, 0.5, 128)
    assert args.run_dir == "/tmp/somewhere/"


def test_run_name_and_paths(parse, utils):
    args = parse("e3c-ner_clinical", "--seed", "1")
    name = utils.create_run_name(args)
    assert re.fullmatch(r"DrBenchmark-e3c-ner_clinical-[0-9a-f]{16}", name)
    assert utils.get_run_path(args) == os.path.join(args.run_dir, f"{name}.json")
    assert utils.get_model_path(args) == os.path.join(args.output_dir, name)


def test_run_name_depends_on_the_arguments(parse, utils):
    reference = utils.create_run_name(parse("cas-pos", "--seed", "1"))
    assert utils.create_run_name(parse("cas-pos", "--seed", "1")) == reference
    assert utils.create_run_name(parse("cas-pos", "--seed", "2")) != reference
    assert utils.create_run_name(parse("cas-pos", "--seed", "1", "--epochs", "3")) != reference
    assert utils.create_run_name(parse("cas-pos", "--seed", "1", model="other/model")) != reference


def test_seed_is_drawn_when_none_is_given(parse):
    seed = parse("cas-pos").seed
    assert isinstance(seed, int) and 0 <= seed < 2**32


def test_run_already_computed_is_skipped(parse, utils, tmp_path):
    arguments = ("--seed", "1", "--run_dir", f"{tmp_path}/")
    run_path = utils.get_run_path(parse("cas-pos", *arguments))
    with open(run_path, "w") as f:
        f.write("{}")

    with pytest.raises(SystemExit) as exit_info:
        parse("cas-pos", *arguments)
    assert exit_info.value.code == 0
    # Without a seed, another one is drawn instead of skipping the run
    assert parse("cas-pos", "--run_dir", f"{tmp_path}/").seed is not None


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_run_already_computed_is_skipped_before_the_libraries_are_loaded(task, parse, utils, tmp_path):
    """Loading torch and transformers takes seconds: the script of a run whose results exist stops before."""
    options = ["--seed", "0", "--run_dir", f"{tmp_path}/"]
    # The results of the run, where its script looks for them
    open(utils.get_run_path(parse(task, *options)), "w").close()

    # The script, as `run.py` starts it
    program, *arguments = task.command(MODEL)
    completed = subprocess.run(
        [program, "-X", "importtime", *arguments, *options], cwd=task.scripts_dir, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.startswith("This configuration was already computed (seed: 0, model: ")
    imported = {line.split("|")[-1].strip().split(".")[0] for line in completed.stderr.splitlines() if line.startswith("import time:")}
    assert "yaml" in imported and not imported & {"torch", "transformers", "datasets", "numpy"}


def test_arguments_of_a_run_are_read_once(parse, utils, monkeypatch):
    """A script reads its arguments before it loads what trains: the training is given them, and does not read others."""
    parse("cas-pos")
    read = []
    parse_args = utils.parse_args
    monkeypatch.setattr(utils, "parse_args", lambda: read.append(parse_args()) or read[-1])

    utils.stop_if_already_computed()
    assert len(read) == 1
    # Without a seed, each reading draws one: the run has to have the first one
    assert utils.arguments_of_the_run() is read[0] and len(read) == 1
    # A script that did not read its arguments first: the training reads them
    assert utils.arguments_of_the_run() is read[1] and len(read) == 2


def test_existing_model_path_is_kept(parse, tmp_path):
    assert parse("cas-pos", model=str(tmp_path)).model_name == str(tmp_path)


def test_model_path_relative_to_the_repository(parse):
    """`run.py` is run from the root of the repository, the scripts from `scripts/`."""
    assert parse("cas-pos", model="assets").model_name == os.path.join("..", "..", "..", "assets")


def test_hub_model_in_offline_mode(parse):
    args = parse("cas-pos", "--offline", "True")
    assert args.offline is True
    assert args.model_name == os.path.join("..", "..", "..", "models", "dr-bert_drbert-7gb")
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_name_of_the_model_on_the_hub_is_kept(parse):
    """In offline mode, `model_name` is the local copy of the model, in lower case: its name is kept apart."""
    offline = parse("cas-pos", "--offline", "True")
    assert offline.model_name != MODEL
    assert offline.hub_model_name == MODEL

    online = parse("cas-pos", "--offline", "False")
    assert (online.model_name, online.hub_model_name) == (MODEL, MODEL)


def test_model_given_by_a_path_has_no_name_on_the_hub(parse, tmp_path):
    assert parse("cas-pos", model=str(tmp_path)).hub_model_name is None
    # A path relative to the root of the repository
    assert parse("cas-pos", model="assets").hub_model_name is None
    # The local copy of a model, given by its path
    (tmp_path / "dr-bert_drbert-7gb").mkdir()
    assert parse("cas-pos", model="dr-bert_drbert-7gb", cwd=tmp_path).hub_model_name is None


def test_name_of_the_model_on_the_hub_is_not_part_of_the_name_of_the_run(parse, utils):
    """The names of the runs must not change: the model is already named by `model_name`."""
    args = parse("cas-pos", "--seed", "1")
    name = utils.create_run_name(args)
    assert "hub_model_name" in vars(args)

    args.hub_model_name = "another/name"
    assert utils.create_run_name(args) == name
    del args.hub_model_name
    assert utils.create_run_name(args) == name
    # The other arguments are part of it
    args.model_name = "another/name"
    assert utils.create_run_name(args) != name


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_arguments_do_not_depend_on_the_starting_directory(task, parse, starting_directories):
    """The name of a run is built from its arguments: they are the same wherever the script is started from."""
    reference = vars(parse(task, "--seed", "1"))
    assert reference["config"] == task.config_arg

    for description, directory in starting_directories(task).items():
        args = parse(task, "--seed", "1", cwd=directory)
        assert vars(args) == reference, description
        # The paths of the arguments are relative to `scripts/`: the script moved there
        assert os.path.samefile(os.getcwd(), task.scripts_dir), description


@pytest.mark.parametrize("spelling", ["./../yaml/pos.yaml", "../yaml/../yaml/pos.yaml", "absolute"])
def test_path_of_the_yaml_file_is_always_written_the_same_way(spelling, parse, monkeypatch, utils):
    """The launchers give `<their directory>/../yaml/<file>.yaml`: the run must keep its name."""
    task = Task("cas-pos")
    reference = vars(parse(task, "--seed", "1"))

    config = str(task.config_path) if spelling == "absolute" else spelling
    monkeypatch.setattr(sys, "argv", [f"./{task.script}", f"--config={config}", f"--model={MODEL}", "--seed", "1"])
    assert vars(utils.parse_args()) == reference


def test_command_line_paths_are_relative_to_the_starting_directory(parse, tmp_path):
    """`--data_dir data/` typed from a directory is the directory `data/` which is there."""
    task = Task("cas-pos")
    typed = ("--data_dir", "data/", "--run_dir", "results", "--output_dir", "models/")
    args = parse(task, "--seed", "1", *typed, cwd=tmp_path)

    # They are rewritten from `scripts/`, the current directory after the parsing
    assert os.path.realpath(args.data_dir) == str(tmp_path / "data")
    assert os.path.realpath(args.run_dir) == str(tmp_path / "results")
    assert os.path.realpath(args.output_dir) == str(tmp_path / "models")
    assert not any(os.path.isabs(path) for path in (args.data_dir, args.run_dir, args.output_dir))
    # Neither the final slash nor its absence are lost: the name of the run depends on them
    assert args.data_dir.endswith("/") and not args.run_dir.endswith("/")

    # The same directories, typed from `scripts/`, give the same arguments
    from_scripts = [relative_path(tmp_path / path, task.scripts_dir) for path in ("data", "results", "models")]
    same = parse(task, "--seed", "1", "--data_dir", from_scripts[0] + "/", "--run_dir", from_scripts[1],
                 "--output_dir", from_scripts[2] + "/")
    assert vars(same) == vars(args)


def test_absolute_paths_are_kept(parse, tmp_path):
    typed = ("--data_dir", "/somewhere/data/", "--run_dir", "/somewhere/runs/", "--output_dir", "/somewhere/models/")
    args = parse("cas-pos", "--seed", "1", *typed, cwd=tmp_path)
    assert (args.data_dir, args.run_dir, args.output_dir) == ("/somewhere/data/", "/somewhere/runs/", "/somewhere/models")
    assert parse("cas-pos", "--seed", "1", *typed) == args


def test_local_model_relative_to_the_starting_directory(parse, tmp_path):
    (tmp_path / "my_model").mkdir()
    args = parse("cas-pos", "--seed", "1", model="my_model", cwd=tmp_path)
    assert not os.path.isabs(args.model_name)
    assert os.path.realpath(args.model_name) == str(tmp_path / "my_model")


@pytest.mark.parametrize("model", ["assets", "assets/"])
def test_local_model_gets_the_same_name_from_every_directory(model, parse, starting_directories):
    """A model of the repository is written `../../../<model>` in the results, as when `run.py` gives it."""
    task = Task("cas-pos")
    reference = parse(task, "--seed", "1", model=model)
    assert reference.model_name == os.path.join("..", "..", "..", model)

    for description, directory in starting_directories(task).items():
        typed = relative_path(REPO / "assets", directory) + ("/" if model.endswith("/") else "")
        assert parse(task, "--seed", "1", model=typed, cwd=directory) == reference, description
        # A path that does not exist from the starting directory is looked for from the root of the repository
        if directory != REPO:
            assert parse(task, "--seed", "1", model=model, cwd=directory) == reference, description


def test_script_which_is_not_the_one_of_a_recipe_is_not_moved(utils, monkeypatch, tmp_path):
    """Out of `recipes/<recipe>/scripts/`, the paths stay relative to the current directory."""
    task = Task("cas-pos")
    script = tmp_path / "script.py"
    script.write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), f"--config={task.config_path}", f"--model={MODEL}", "--seed", "1"])

    args = utils.parse_args()
    assert os.path.samefile(os.getcwd(), tmp_path)
    assert args.config == str(task.config_path)
    assert args.run_dir == task.config["run_dir"]


def test_fewshot_and_max_train_samples_are_exclusive(parse):
    with pytest.raises(ValueError):
        parse("cas-pos", "--fewshot", "0.5", "--max_train_samples", "10")
    assert parse("cas-pos", "--max_train_samples", "10").max_train_samples == 10


GRADIENT_ACCUMULATION_TASKS = [task for task in ALL_TASKS if "gradient_accumulation_steps" in task.config]


def test_some_tasks_accumulate_gradients():
    """The tests below would check nothing if no YAML file set `gradient_accumulation_steps` any more."""
    assert {task.name for task in GRADIENT_ACCUMULATION_TASKS} == {"diamed-cls", "morfitt-cls", "deft2021-ner", "deft2021-cls"}


@pytest.mark.parametrize("task", GRADIENT_ACCUMULATION_TASKS, ids=task_ids)
def test_gradient_accumulation_of_the_yaml_file_is_kept(task, parse):
    assert task.config["gradient_accumulation_steps"] > 1
    assert parse(task).gradient_accumulation_steps == task.config["gradient_accumulation_steps"]


def test_gradient_accumulation_on_the_command_line_overrides_the_yaml_file(parse):
    assert parse("diamed-cls", "--gradient_accumulation_steps", "3").gradient_accumulation_steps == 3
    assert parse("cas-pos", "--gradient_accumulation_steps", "3").gradient_accumulation_steps == 3


@pytest.mark.parametrize(
    "task", [task for task in ALL_TASKS if task not in GRADIENT_ACCUMULATION_TASKS], ids=task_ids,
)
def test_no_gradient_accumulation_by_default(task, parse):
    """Without accumulation, the weights are updated after each batch."""
    assert parse(task).gradient_accumulation_steps == 1


@pytest.mark.parametrize("value", ["0", "-2"])
def test_gradient_accumulation_needs_at_least_one_batch(value, parse):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        parse("cas-pos", "--gradient_accumulation_steps", value)


@pytest.mark.parametrize("value, expected", [(None, True), ("true", True), ("false", False)])
def test_model_of_a_run_is_kept_unless_it_is_asked_not_to(value, expected, parse):
    args = parse("cas-pos", "--seed", "1", *(["--keep_model", value] if value else []))
    assert args.keep_model is expected


def test_what_is_kept_of_a_run_is_not_part_of_its_name(parse, utils, tmp_path):
    """A run is the same one with or without its model: it is not computed twice."""
    kept = parse("cas-pos", "--seed", "1", "--run_dir", f"{tmp_path}/")
    not_kept = parse("cas-pos", "--seed", "1", "--run_dir", f"{tmp_path}/", "--keep_model", "false")
    assert kept.keep_model != not_kept.keep_model
    assert utils.create_run_name(kept) == utils.create_run_name(not_kept)
    assert utils.get_run_path(kept) == utils.get_run_path(not_kept)

    # Once it was computed without its model, it is not computed again to keep it
    open(utils.get_run_path(not_kept), "w").close()
    with pytest.raises(SystemExit) as exit_info:
        parse("cas-pos", "--seed", "1", "--run_dir", f"{tmp_path}/")
    assert exit_info.value.code == 0


@pytest.mark.parametrize("task", ALL_TASKS, ids=task_ids)
def test_run_of_run_py_is_the_one_of_the_launcher(task, parse, utils, monkeypatch):
    """`run.py` starts the script of a task itself: its run has the arguments, hence the name, of the one of the launcher.

    The runs that the launchers computed are the ones that `run.py` asks for: they are not computed again.
    """
    by_the_launcher = parse(task, "--seed", "3")
    monkeypatch.chdir(task.scripts_dir)
    # The command of `run.py`, without its program
    monkeypatch.setattr(sys, "argv", [*task.command(MODEL)[1:], "--seed", "3"])
    by_run_py = utils.parse_args()
    assert by_run_py == by_the_launcher
    assert utils.get_run_path(by_run_py) == utils.get_run_path(by_the_launcher)


def test_name_of_a_run_is_stable(parse, utils):
    """An unchanged configuration keeps the name of its run from one version of the code to the next.

    This is how a run already computed is recognised. The name is built from every argument:
    if this test fails, the change gives a new name to all the runs, even to the ones it does
    not affect. The values of `config.yaml` are given, for the test not to depend on that file.
    """
    args = parse("cas-pos", "--seed", "0", "--offline", "True", "--max_position_embeddings", "512")
    assert utils.create_run_name(args) == "DrBenchmark-cas-pos-8c05112b3a9ba5af"


@pytest.mark.parametrize("value, expected", [
    ("True", True), ("true", True), ("yes", True), ("1", True),
    ("False", False), ("false", False), ("no", False), ("0", False),
])
def test_offline_mode_on_the_command_line(value, expected, parse, monkeypatch):
    """`--offline` overrides `config.yaml`, in both directions."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    args = parse("cas-pos", "--offline", value)
    assert args.offline is expected
    assert ("HF_HUB_OFFLINE" in os.environ) is expected
    if not expected:
        # The model is downloaded from the Hub: its name is kept as it is
        assert args.model_name == MODEL


def test_offline_mode_needs_a_boolean(parse, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse("cas-pos", "--offline", "maybe")
    assert exit_info.value.code == 2
    assert "expected true or false" in capsys.readouterr().err
