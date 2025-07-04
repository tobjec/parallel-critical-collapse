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

def parse_json(input_path: Path=INPUT_FILE) -> dict:
    """
    Function to parse simulation input creation json file to dictionary.

    Args:
        file_path (Path): Path to the input file

    Returns:
        dict: Dictionary of the simulation create data
    """
    with open(input_path.as_posix(), 'r') as f:
        return json.load(f)

def create_sim_input(sim_dict: dict, output_path: Path=OUTPUT_FILE) -> None:
    """
    Routine to create the final simulation data file.

    Args:
        sim_dict (dict): Parsed json file to dict.
        output_path (Path, optional): Path to the output file. Defaults to OUTPUT_FILE.
    """
    overall_dict = OrderedDict()
    keys = sorted(list(sim_dict.keys()))
    first_dim = keys.pop()
    first_dict = sim_dict[first_dim]
    overall_dict[first_dict["Dim"]] = first_dict

    for key in keys:
        input_dict = sim_dict[key]
        first_range, last_range = input_dict["Dims"]
        delta_dims = input_dict["dDims"]
        decimal_places = abs(Decimal(str(delta_dims)).as_tuple().exponent)
        dims = np.round(np.append(np.arange(first_range, last_range, delta_dims, dtype=np.float64), last_range), decimal_places)
        for dim in dims:
            base_dict = deepcopy(input_dict["SimInput"])
            base_dict.update({"Dim": dim, "Initial_Condition": {"Delta": 0, "fc": [], "psic": [], "Up":[]}})
            overall_dict[round(dim,3)] = base_dict
    
    with open(output_path.as_posix(), "w") as f:
        json.dump(overall_dict, f)

# ============================ 4. Main Calculations ==============================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate cricital collapse simulation input JSON from a template config.'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default="../data/simulation_config.json",
        help='Path to the input JSON file.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="../data/simulation_data.json",
        help='Path to save the generated simulation data.'
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    sim_config = parse_json(input_path)
    create_sim_input(sim_config, output_path)