"""CogScale data generation and validated preset overrides."""
from copy import deepcopy
from typing import Final, Tuple, Optional, Dict, Iterable, Any
import cogscale as cog


TASKS: Final[Tuple] = ("simple_copy", "selective_copy", "induction_heads", "associative_recall", "adding_problem", "sorting_problem")
TASK_SIZE_OPTIONS: Final[Tuple] = ("sequence_length", "n_symbols", "delay", "n_markers", "num_pairs")


def resolve_task_sizes(task: str, difficulty: str, overrides: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """
    Resolve and validate task sizes without changing library presets.

    Args:
        task: Supported CogScale task name.
        difficulty: Library preset: small, medium, or large.
        overrides: Optional mapping of size names to integers. None retains the preset; options for other tasks are
            ignored.

    Returns:
        Dictionary of effective size parameters applicable to this task.

    Raises:
        ValueError: An override is unknown or effective sizes are invalid.
    """
    params = deepcopy(getattr(cog.evals, "cogscale_" + difficulty)[task]["params"])
    for name, value in (overrides or {}).items():
        if name not in TASK_SIZE_OPTIONS:
            raise ValueError(f"Unknown task size option: {name}")
        if value is not None and name in params:
            params[name] = value
    sizes = {name: params[name] for name in TASK_SIZE_OPTIONS if name in params}
    for name, value in sizes.items():
        minimum = 0 if name == "delay" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{task}: {name} must be an integer >= {minimum}.")
    if task == "selective_copy" and sizes["n_markers"] > sizes["sequence_length"]:
        raise ValueError("selective_copy: n_markers must not exceed sequence_length.")
    if task == "induction_heads" and (sizes["sequence_length"] < 4
                                      or sizes["sequence_length"] % 2):
        raise ValueError("induction_heads: sequence_length must be even and >= 4.")
    if task == "associative_recall":
        if sizes["sequence_length"] < 2 * sizes["num_pairs"] + 2:
            raise ValueError("associative_recall: sequence_length must be >= 2*num_pairs + 2.")
        if sizes["n_symbols"] < sizes["num_pairs"] + 2:
            raise ValueError("associative_recall: n_symbols must be >= num_pairs + 2 "
                             "(two reserved symbols).")
    return sizes


def resolve_task_overrides(tasks: Iterable[str],
                           difficulty: str,
                           overrides: Optional[Dict[str, int]] = None) -> Dict[str, Dict[str, int]]:
    """
    Validate overrides across all selected tasks before generating data.

    Args:
        tasks: Iterable of supported task names.
        difficulty: Library difficulty preset.
        overrides: Optional mapping of size names to integers or None.

    Returns:
        Mapping from each selected task to its effective size parameters.

    Raises:
        ValueError: Sizes are invalid or a supplied override applies to no task.
    """
    sizes = {task: resolve_task_sizes(task, difficulty, overrides) for task in tasks}
    for name, value in (overrides or {}).items():
        if value is not None and not any(name in values for values in sizes.values()):
            raise ValueError(f"--{name} does not apply to any selected task.")
    return sizes


def build_data(task: str,
               difficulty: str,
               seed: int,
               n_train: int,
               n_val: int,
               n_test: int,
               *,
               overrides: Optional[Dict[str, int]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate fixed classification splits while restoring library presets.

    Args:
        task: Supported CogScale task name.
        difficulty: Library preset: small, medium, or large.
        seed: Seed passed unchanged to CogScale.
        n_train: Number of training examples.
        n_val: Number of validation examples.
        n_test: Number of test examples, generated even when testing is skipped.
        overrides: Optional size overrides; None values retain preset sizes.

    Returns:
        Pair of the library data dictionary and a copy of the effective task parameters. Data includes X/Y arrays for
        train, valid, and test splits.

    Raises:
        ValueError: Sizes are invalid or the task is not classification.
    """
    # CogScale 0.2.0 mutates its preset dictionary in build_task. Restore it so repeated calls in the same process
    # cannot contaminate subsequent tasks.
    registry = getattr(cog.evals, "cogscale_" + difficulty)
    original = deepcopy(registry[task]["params"])
    sizes = resolve_task_sizes(task, difficulty, overrides)
    try:
        data = cog.build_task(task, difficulty=difficulty, seed=seed,
                              n_train=n_train, n_valid=n_val, n_test=n_test, **sizes)
        params = deepcopy(registry[task]["params"])
    finally:
        registry[task]["params"] = original
    if data["category"] != "classification":
        raise ValueError("This runner supports CogScale classification tasks only.")
    return data, params

