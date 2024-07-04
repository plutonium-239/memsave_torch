"""Utility class to collect the results of estimation and make a dataframe from all the models and cases."""

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Callable, List, Union

import pandas as pd
from tqdm import tqdm

strings = {
    "time": [
        "s",
        "T",
        "speed-up",
        "faster",
        "Estimated time speed-up",
        "Time Taken (s)",
    ],
    "memory": [
        "MB",
        "M",
        "savings",
        "memory",
        "Estimated memory savings",
        "Memory Usage (MB)",
    ],
}

cases = {
    "All": None,  # ALL
    "Input": [  # INPUT
        "grad_input",
        "no_grad_conv_weights",
        "no_grad_conv_bias",
        "no_grad_linear_weights",
        "no_grad_linear_bias",
        "no_grad_norm_weights",
        "no_grad_norm_bias",
    ],
    "Everything": [  # INPUT
        "grad_input"
    ],
    "Nothing": [
        "no_grad_conv_weights",
        "no_grad_conv_bias",
        "no_grad_linear_weights",
        "no_grad_linear_bias",
        "no_grad_norm_weights",
        "no_grad_norm_bias",
    ],
    "Conv": [  # CONV
        "no_grad_linear_weights",
        "no_grad_linear_bias",
        "no_grad_norm_weights",
        "no_grad_norm_bias",
    ],
    "Linear": [  # LINEAR
        "no_grad_conv_weights",
        "no_grad_conv_bias",
        "no_grad_norm_weights",
        "no_grad_norm_bias",
    ],
    "Norm": [  # NORM
        "no_grad_conv_weights",
        "no_grad_conv_bias",
        "no_grad_linear_weights",
        "no_grad_linear_bias",
    ],
    "SurgicalFirst": ["surgical_first"],
    "SurgicalLast": ["surgical_last", "grad_input"],
}


def select_cases(selected: List[str]) -> List[Union[List[str], None]]:
    """Helper function to return cases selected by their names

    Args:
        selected (List[str]): Which cases to select, strings can be keys of the cases table

    Returns:
        List[Union[List[str], None]]: Selected cases
    """
    return [cases[s] for s in selected]


def make_case_str(case: Union[None, List[str]]) -> str:
    """Format case into a string

    Args:
        case (Union[None, List[str]]): Given case

    Returns:
        str: Output
    """
    return "None" if case is None else " + ".join(case)


case_inv_mapping = {make_case_str(v): k for k, v in cases.items()}


def hyperparam_str(args: SimpleNamespace) -> str:
    """Format hyperparams into a string

    Args:
        args: args

    Returns:
        str: Output
    """
    return f"HW={args.input_hw} B={args.batch_size} C_in={args.input_channels}"


class ResultsCollector:
    """This class collects results by reading from the results/ directory"""

    def __init__(
        self,
        batch_size: int,
        input_channels: int,
        input_HW: int,
        num_classes: int,
        device: str,
        architecture: str,
        vjp_improvements: List[float],
        cases: List[Union[None, List[str]]],
        results_dir: str,
        print: Callable = tqdm.write,
    ) -> None:
        """Initialize the collector before all runs.

        Args:
            batch_size (int): batch_size
            input_channels (int): input_channels
            input_HW (int): input_HW
            num_classes (int): num_classes
            device (str): device
            architecture (str): conv or linear
            vjp_improvements (List[float]): vjp_improvements
            cases (List[Union[None, List[str]]]): list of cases
            results_dir (str): The base results dir
            print (Callable, optional): Which function to use for printing (i.e. `print()` causes problems in a tqdm context)
        """
        # TODO: architecture is pointless since there is no arch-specific code anymore
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.input_HW = input_HW
        self.num_classes = num_classes
        self.device = device
        self.architecture = architecture
        self.vjp_improvements = vjp_improvements
        self.cases = cases
        # assert len(cases) == 3, f"len(cases) > 3:\n{cases}" Not anymore
        self.base_location = os.path.join(results_dir, "raw")
        os.makedirs(self.base_location, exist_ok=True)
        self.savings = pd.DataFrame(
            columns=["model", "input_vjps", strings["time"][4], strings["memory"][4]]
        )
        self.usage_stats = pd.DataFrame(
            columns=["model", "case", strings["time"][5], strings["memory"][5]]
        )
        self.savings.set_index(["model", "input_vjps"], inplace=True)
        self.usage_stats.set_index(["model", "case"], inplace=True)
        self.print = print

    def collect_from_file(self, estimate: str, model: str):
        """To be called after all cases of a model have finished.

        Args:
            estimate (str): time or memory
            model (str): The name of the model

        Raises:
            AssertionError: If the temp file has more/less lines than expected (= number of cases)
            ValueError: If the temp file has unallowed text (only floats are allowed)
        """
        with open(f"{self.base_location}/{estimate}-{self.architecture}.txt") as f:
            lines = f.readlines()

        try:
            assert (
                len(lines) == len(self.cases)
            ), f"More than {len(self.cases)} lines found in {self.base_location}/{estimate}-{self.architecture}.txt:\n{lines}"
            outputs = [float(line.strip()) for line in lines]
            for case, out in zip(self.cases, outputs):
                self.usage_stats.loc[
                    (model, make_case_str(case)), strings[estimate][5]
                ] = out

            self._display_run(outputs, estimate, model, self.print)
        except AssertionError as e:
            raise e
        except ValueError as e:
            print(
                f'File {self.base_location}/{estimate}-{self.architecture}.txt has unallowed text. Contents: \n{"".join(lines)}'
            )
            raise e
        finally:
            self.clear_file(estimate)

    def clear_file(self, estimate: str):
        """Clears the temp file for the given estimation, to be called before and after collecting stats for a run.

        Args:
            estimate (str): time or memory
        """
        with open(f"{self.base_location}/{estimate}-{self.architecture}.txt", "w") as f:
            f.write("")

    def _display_run(
        self,
        outputs: List[float],
        estimate: str,
        model: str,
        print: Callable,
    ):
        """Function to display the data collected over all cases for a model.

        Args:
            outputs (List[float]): The collected outputs
            estimate (str): time or memory
            model (str): The name of the model
            print (Callable): Which function to use for printing (i.e. `print()` causes problems in a tqdm context)
        """
        # print(f"{model} input ({input_channels},{input_HW},{input_HW}) {device}")
        # print('='*78)
        if self.architecture == "conv":
            s = f"{model} input ({self.batch_size},{self.input_channels},{self.input_HW},{self.input_HW}) {self.device}"
        elif self.architecture == "transformer":
            s = f"{model} input ({self.batch_size},{self.input_HW},{self.input_channels}(or model hidden size)) {self.device}"
        elif self.architecture == "linear":
            s = f"{model} input ({self.batch_size},{self.input_HW**2}) {self.device}"
        print(s.center(78, "="))

        for out, case in zip(outputs, self.cases):
            print(
                f"{strings[estimate][1]} ({case_inv_mapping[make_case_str(case)]}): {out:.3f}{strings[estimate][0]}"
            )

        # CODE ONLY APPLIES WITH OLD RUNDEMO.PY
        # q_conv_weight = outputs[1] - outputs[2]
        # ratio = q_conv_weight / outputs[0]
        # if estimate == "time":
        #     print(
        #         f"{self.architecture.capitalize()} weight VJPs use {100 * ratio:.1f}% of time"
        #     )
        # else:
        #     print(
        #         f"Information for {self.architecture} weight VJPs uses {100 * ratio:.1f}% of memory"
        #     )
        # # self.models.loc[model, '']

        # tot_improvements = [
        #     1 - (1 - improvement) * ratio for improvement in self.vjp_improvements
        # ]
        # for vjp, tot in zip(self.vjp_improvements, tot_improvements):
        #     print(
        #         f"Weight VJP {strings[estimate][2]} of {vjp:.2f}x ({1 / vjp:.1f}x {strings[estimate][3]})"
        #         + f" would lead to total {strings[estimate][2]} of {tot:.2f}x ({1 / tot:.1f}x {strings[estimate][3]})"
        #     )
        #     self.savings.loc[(model, vjp), strings[estimate][4]] = f"{1 / tot:.1f}x"
        print("")

    def finish(self):
        """To be called after ALL cases on all models have been run, saves dataframes to csv files."""
        time = datetime.now().strftime("%d.%m.%y %H.%M")
        s = f"input ({self.batch_size},{self.input_channels},{self.input_HW},{self.input_HW}) {self.device}"
        # savings_path = f"{self.base_location}/../savings-{self.architecture}-{self.device}-{time}.csv"
        # with open(savings_path, "w") as f:
        #     f.write(s + "\n")
        # self.savings.to_csv(savings_path, mode="a")

        usage_path = f"{self.base_location}/../usage_stats-{self.architecture}-{self.device}-{time}.csv"
        with open(usage_path, "w") as f:
            f.write(s + "\n")
        self.usage_stats.to_csv(usage_path, mode="a")
