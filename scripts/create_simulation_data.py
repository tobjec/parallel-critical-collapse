#!/bin/usr/python3

# ============================ 1. Importing Libraries ============================

import numpy as np
import json
from decimal import Decimal
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy
import argparse

# ============================ 2. Defining Parameters ============================

INPUT_FILE_PATH: str = "../data/simulation_config.json"
INPUT_FILE: Path = Path(INPUT_FILE_PATH)
OUTPUT_FILE_PATH: str = "../data/simulation_data.json"
OUTPUT_FILE: Path = Path(OUTPUT_FILE_PATH)

# ============================ 3. Defining Functions =============================


def parse_json(input_path: Path = INPUT_FILE) -> OrderedDict:
    """
    Function to parse simulation input creation json file to dictionary.

    Args:
        file_path (Path): Path to the input file.

    Returns:
        OrderedDict: Ordered dictionary of the simulation create data.
    """
    with open(input_path.as_posix(), "r") as f:
        return OrderedDict(json.load(f))


def double(data: list) -> np.ndarray:
    """
    Computing of the doubled amount of time points of initial data

    Args:
        data (list): Input initial data.

    Returns:
        np.ndarray: Doubled initial data.
    """
    fourier = np.fft.fft(data)
    nt = 2 * int(len(data))
    doubled = [0] * nt

    doubled[: int(nt / 4 - 1)] = fourier[: int(nt / 4 - 1)]
    doubled[int(nt / 4)] = fourier[int(nt / 4)] / 2
    doubled[int(3 * nt / 4)] = fourier[int(nt / 4)] / 2
    doubled[int(3 * nt / 4 + 1) :] = fourier[int(nt / 4 + 1) :]

    modified_data = np.sqrt(4) * np.fft.ifft(doubled)

    return np.real(modified_data)


def create_sim_input(
    sim_dict: dict, output_path: Path = OUTPUT_FILE, reversed: bool = False
) -> None:
    """
    Function to create the final simulation data file.

    Args:
        sim_dict (dict): Parsed json file to dict.
        output_path (Path, optional): Path to the output file. Defaults to OUTPUT_FILE.
        reversed (bool, optional): Reverse the order of the dimensions. Defaults to False.
    """
    overall_dict = OrderedDict()
    keys = sorted(list(sim_dict.keys()))
    first_dim = keys.pop()
    first_dict = sim_dict[first_dim]
    overall_dict[str(first_dict["Dim"])] = first_dict

    for key in keys:
        input_dict = sim_dict[key]
        first_range, last_range = input_dict["Dims"]
        delta_dims = input_dict["dDims"]
        decimal_places = abs(Decimal(str(delta_dims)).as_tuple().exponent)
        dims = np.round(
            np.append(
                np.arange(first_range, last_range, delta_dims, dtype=np.float64),
                last_range,
            ),
            decimal_places,
        )
        for dim in dims:
            base_dict = deepcopy(input_dict["SimInput"])
            base_dict.update(
                {
                    "Dim": dim,
                    "Initial_Conditions": {"Delta": 0, "fc": [], "psic": [], "Up": []},
                }
            )
            overall_dict[f"{dim}"] = base_dict

    if reversed:
        new_overall_dict = OrderedDict()
        reversed_keys = sorted(overall_dict.keys())
        for key in reversed_keys:
            new_overall_dict[key] = overall_dict[key]
        with open(output_path.as_posix(), "w") as f:
            json.dump(new_overall_dict, f)
    else:
        with open(output_path.as_posix(), "w") as f:
            json.dump(overall_dict, f)


def create_doub_input(
    sim_dict: dict,
    output_path: Path = INPUT_FILE,
    dim: str = None,
    reversed: bool = False,
) -> None:
    """
    Function to create the doubled simulation initial data input

    Args:
        sim_dict (dict): Dictionary of simulation data.
        output_path (Path, optional): Path to output. Defaults to INPUT_FILE.
        dim (str, optional): Dimension if multidimensional input file. Defaults to None.
        reversed (bool, optional): Reversed order of remaining doubled input data. Defaults to None.
    """

    doubled_dict = sim_dict

    if dim:
        dim_dict = list(
            filter(lambda x: np.isclose(float(x), float(dim)), list(sim_dict.keys()))
        )[0]
        doubled_dict_dim = doubled_dict[dim_dict]
    else:
        doubled_dict_dim = doubled_dict

    fc_doub = double(doubled_dict_dim["Initial_Conditions"]["fc"])
    psic_doub = double(doubled_dict_dim["Initial_Conditions"]["psic"])
    Up_doub = double(doubled_dict_dim["Initial_Conditions"]["Up"])

    doubled_dict_dim["Ntau"] = len(fc_doub)
    doubled_dict_dim["Converged"] = False
    doubled_dict_dim["Initial_Conditions"]["fc"] = fc_doub.tolist()
    doubled_dict_dim["Initial_Conditions"]["psic"] = psic_doub.tolist()
    doubled_dict_dim["Initial_Conditions"]["Up"] = Up_doub.tolist()

    if dim:
        doubled_dict[dim_dict] = doubled_dict_dim
        for key in doubled_dict.keys():
            if float(key) < float(dim_dict) and not reversed:
                doubled_dict[key]["Ntau"] = len(fc_doub)
                doubled_dict_dim["Converged"] = False
            elif float(key) > float(dim_dict) and reversed:
                doubled_dict[key]["Ntau"] = len(fc_doub)
                doubled_dict_dim["Converged"] = False
    else:
        doubled_dict = doubled_dict_dim

    with open(output_path.as_posix(), "w") as f:
        json.dump(doubled_dict, f)


def merge(
    input_paths: list, output_path: Path = OUTPUT_FILE, reversed: bool = False
) -> None:
    """
    Routine to merge multidimensional simulation files.

    Args:
        input_paths (list): List of paths to the simulation files.
        output_path (Path, optional): Name of output file. Defaults to OUTPUT_FILE.
        reversed (bool, optional): Reverse the ordering. Defaults to False.
    """
    overall_dict = OrderedDict()
    tmp_dict = dict()

    for input_path in input_paths:
        tmp_dict.update(parse_json(Path(input_path)))

    sorted_keys = sorted(tmp_dict.keys(), reverse=reversed)

    for key in sorted_keys:
        overall_dict[key] = tmp_dict[key]

    with open(output_path.as_posix(), "w") as f:
        json.dump(overall_dict, f)


# ============================ 4. Main Calculations ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cricital collapse simulation input JSON from template configuration."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        default=["../data/simulation_config.json"],
        help="Path to the input JSON file(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="+",
        default=["../data/simulation_data.json"],
        help="Path to save the generated simulation data file(s).",
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        choices=["create_multidim", "double_nt", "merge"],
        default="create_multidim",
        help="Kind of operation to be conducted.",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=str,
        default="",
        help="Dimension of the particular simulation.",
    )
    parser.add_argument(
        "-r",
        "--reversed",
        action="store_true",
        help="Sets reversed order of higher dimensions when updating Ntau.",
    )
    args = parser.parse_args()

    if len(args.input) == 1 and len(args.output) == 1:
        input_path = Path(args.input[0])
        output_path = Path(args.output[0])
        match args.kind:
            case "create_multidim":
                create_sim_input(parse_json(input_path), output_path, args.reversed)
            case "double_nt":
                create_doub_input(
                    parse_json(input_path), output_path, args.dim, args.reversed
                )
            case _:
                raise ValueError(
                    f"{args.kind} is not a valid option for the"
                    + " given input and output files."
                )
    elif len(args.input) > 1 and len(args.output) > 1:
        for input, output in zip(args.input, args.output):
            input_path = Path(input)
            output_path = Path(output)
            match args.kind:
                case "create_multidim":
                    create_sim_input(parse_json(input_path), output_path, args.reversed)
                case "double_nt":
                    create_doub_input(
                        parse_json(input_path), output_path, args.dim, args.reversed
                    )
                case _:
                    raise ValueError(
                        f"{args.kind} is not a valid option for the"
                        + " given input and output files."
                    )
    elif len(args.input) > 1 and len(args.output) == 1:
        merge(args.input, Path(args.output[0]), args.reversed)
    else:
        raise IndexError("Invalid input and output file combo.")
