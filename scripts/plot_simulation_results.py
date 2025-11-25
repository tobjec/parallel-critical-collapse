#!/bin/usr/python3

# ============================ 1. Importing Libraries ============================

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LightSource
from itertools import product
from scipy.interpolate import CubicSpline
from glob import glob
from pathlib import Path
import argparse

# ============================ 2. Defining Parameters ============================

plt.rc("xtick", labelsize=22)
plt.rc("ytick", labelsize=22)
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["ytick.minor.width"] = 1.5
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["ytick.minor.size"] = 3

# Activate LaTeX style if supported
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# ============================ 3. Defining Functions =============================


def round_up(val: float) -> float:
    """
    Round up a float to the next "nice" power-of-10-aligned multiple.

    Args:
        val (float): Input float value.

    Returns:
        float: Output float value.
    """

    if val == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(val))))
    base = 10**exponent
    return np.ceil(val / base) * base


def round_down(val: float) -> float:
    """
    Round down a float to the next "nice" power-of-10-aligned multiple.

    Args:
        val (float): Input float value.

    Returns:
        float: Output float value.
    """

    if val == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(val))))
    base = 10**exponent
    return np.floor(val / base) * base


def round_limits(zmin: float, zmax: float) -> tuple:
    """
    Round zmin down and zmax up to nearest power-of-10-aligned "nice" bounds.

    Args:
        zmin (float): Input lower limit.
        zmax (float): Input upper limit.

    Returns:
        tuple: Output tuple of lower and upper limit.
    """
    return round_down(zmin), round_up(zmax)


# ============================ 4. Defining Classes ===============================


class ResultPlotter:
    """
    Class to plot results of critical collapse simulation.
    """

    def __init__(self, input_files: list = []):
        self.input_files = input_files

        for i, file in enumerate(self.input_files):
            self.input_files[i] = Path(file).absolute()

    def plot(
        self,
        kind: str,
        save_fpath: str = None,
        single_plots: bool = False,
        dim: str | float = None,
        experimental_data: str = None,
        spec: str = None,
    ) -> None:
        """
        Method for creating plots of simulation.

        Args:
            kind (str): Kind of the plot.
            save_fpath (str, optional): Path to save file. Defaults to None.
            single_plots (bool, optional): Creating single plots. Defaults to False.
            dim (str | float, optional): Dimension to be plotted. Defaults to None.
            experimental_data (str, optional): Data to be plotted as reference. Defaults to None.
            spec (str, optional): Plotting kind specification. Defaults to None.

        Raises:
            ValueError: In case plotting kind is not valid.
        """

        if save_fpath:
            save_fpath = Path(save_fpath).absolute()
            if not save_fpath.exists():
                save_fpath.parent.mkdir(parents=True, exist_ok=True)

        match kind:
            case "convergence":
                self.plot_convergence(save_fpath, single_plots, spec)
            case "fields":
                self.plot_fields(save_fpath, single_plots)
            case "fields_paper":
                self.plot_fields_paper(save_fpath, single_plots)
            case "mismatch_layer_finder":
                self.plot_mismatch_pos_finder(save_fpath, spec)
            case "initial_data":
                self.plot_initial_data(save_fpath, dim, experimental_data, spec)
            case "echoing_period":
                self.plot_echoing_period(save_fpath, experimental_data, spec)
            case "theoretical_speedup":
                self.plot_max_speedup(save_fpath, dim)
            case "benchmark":
                self.plot_benchmark(save_fpath, dim, bool(spec))
            case "efficiency":
                self.plot_efficiency(save_fpath, dim, bool(spec))
            case _:
                raise ValueError(f"{kind} is not a valid keyword for creating a plot.")

    def plot_convergence(self, save_fpath: Path, single_plots: bool, spec: str) -> None:

        assert len(self.input_files) == 5, (
            "There should be 5 files supplied to create" + " the convergence plot."
        )

        result_dict = {}

        for file in self.input_files:
            name_list = file.name.split(".")[0].split("_")
            name = name_list[-2] if spec else name_list[-1]
            with open(file.as_posix(), "r") as f:
                result_dict[name] = json.load(f)
            assert result_dict[name][
                "Converged"
            ], f"{name} simulation is not converged, plots cannot be produced."
            result_dict[name]["tau"] = np.linspace(
                0,
                result_dict[name]["Initial_Conditions"]["Delta"],
                num=result_dict[name]["Ntau"],
            )

        dim = result_dict["base"]["Dim"]
        fac = -3 if not spec else -4

        if single_plots:

            for side, field in product(["left", "right"], ["fc", "psic", "Up"]):

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 10))
                tau = np.array(result_dict["base"]["tau"])
                modes = np.fft.rfftfreq(tau.size, d=tau[1] - tau[0])[
                    : int(tau.size // 4)
                ]

                if side == "left":

                    match field:

                        case "fc":
                            field1 = np.array(
                                result_dict["xcut02"]["Initial_Conditions"]["fc"]
                            ) - np.array(
                                result_dict["base"]["Initial_Conditions"]["fc"]
                            )
                            field2 = 2 ** (-6) * (
                                np.array(
                                    result_dict["xcut04"]["Initial_Conditions"]["fc"]
                                )
                                - np.array(
                                    result_dict["xcut02"]["Initial_Conditions"]["fc"]
                                )
                            )
                            field1_k = np.abs(
                                np.fft.rfft(
                                    result_dict["xcut02"]["Initial_Conditions"]["fc"]
                                )
                                - np.fft.rfft(
                                    result_dict["base"]["Initial_Conditions"]["fc"]
                                )
                            )[: int(tau.size // 4)]
                            field2_k = (
                                2 ** (-6)
                                * np.abs(
                                    np.fft.rfft(
                                        result_dict["xcut04"]["Initial_Conditions"][
                                            "fc"
                                        ]
                                    )
                                    - np.fft.rfft(
                                        result_dict["xcut02"]["Initial_Conditions"][
                                            "fc"
                                        ]
                                    )
                                )[: int(tau.size // 4)]
                            )
                            label1 = r"$f_c(2x_L) - f_c(x_L)$"
                            label2 = r"$2^{-6} \left[f_c(4x_L) - f_c(2x_L) \right]$"
                            label1_k = (
                                r"$\left|\hat{f}^k_c(2x_L) - \hat{f}^k_c(x_L)\right|$"
                            )
                            label2_k = r"$2^{-6} \left|\hat{f}^k_c(4x_L) - \hat{f}^k_c(2x_L) \right|$"
                        case "psic":
                            field1 = np.array(
                                result_dict["xcut02"]["Initial_Conditions"]["psic"]
                            ) - np.array(
                                result_dict["base"]["Initial_Conditions"]["psic"]
                            )
                            field2 = 2 ** (-6) * (
                                np.array(
                                    result_dict["xcut04"]["Initial_Conditions"]["psic"]
                                )
                                - np.array(
                                    result_dict["xcut02"]["Initial_Conditions"]["psic"]
                                )
                            )
                            field1_k = np.abs(
                                np.fft.rfft(
                                    result_dict["xcut02"]["Initial_Conditions"]["psic"]
                                )
                                - np.fft.rfft(
                                    result_dict["base"]["Initial_Conditions"]["psic"]
                                )
                            )[: int(tau.size // 4)]
                            field2_k = (
                                2 ** (-6)
                                * np.abs(
                                    np.fft.rfft(
                                        result_dict["xcut04"]["Initial_Conditions"][
                                            "psic"
                                        ]
                                    )
                                    - np.fft.rfft(
                                        result_dict["xcut02"]["Initial_Conditions"][
                                            "psic"
                                        ]
                                    )
                                )[: int(tau.size // 4)]
                            )
                            label1 = r"$\psi_c(2x_L) - \psi_c(x_L)$"
                            label2 = (
                                r"$2^{-6} \left[\psi_c(4x_L) - \psi_c(2x_L) \right]$"
                            )
                            label1_k = r"$\left|\hat{\psi}^k_c(2x_L) - \hat{\psi}^k_c(x_L)\right|$"
                            label2_k = r"$2^{-6} \left|\hat{\psi}^k_c(4x_L) - \hat{\psi}^k_c(2x_L) \right|$"
                        case "Up":
                            field1 = np.array(
                                result_dict["xcut02"]["Initial_Conditions"]["Up"]
                            ) - np.array(
                                result_dict["base"]["Initial_Conditions"]["Up"]
                            )
                            field2 = 2 ** (-6) * (
                                np.array(
                                    result_dict["xcut04"]["Initial_Conditions"]["Up"]
                                )
                                - np.array(
                                    result_dict["xcut02"]["Initial_Conditions"]["Up"]
                                )
                            )
                            field1_k = np.abs(
                                np.fft.rfft(
                                    result_dict["xcut02"]["Initial_Conditions"]["Up"]
                                )
                                - np.fft.rfft(
                                    result_dict["base"]["Initial_Conditions"]["Up"]
                                )
                            )[: int(tau.size // 4)]
                            field2_k = (
                                2 ** (-6)
                                * np.abs(
                                    np.fft.rfft(
                                        result_dict["xcut04"]["Initial_Conditions"][
                                            "Up"
                                        ]
                                    )
                                    - np.fft.rfft(
                                        result_dict["xcut02"]["Initial_Conditions"][
                                            "Up"
                                        ]
                                    )
                                )[: int(tau.size // 4)]
                            )
                            label1 = r"$U_p(2x_L) - U_p(x_L)$"
                            label2 = r"$2^{-6} \left[U_p(4x_L) - U_p(2x_L) \right]$"
                            label1_k = (
                                r"$\left|\hat{U}^k_p(2x_L) - \hat{U}^k_p(x_L)\right|$"
                            )
                            label2_k = r"$2^{-6} \left|\hat{U}^k_p(4x_L) - \hat{U}^k_p(2x_L) \right|$"

                    axes[0].plot(tau, field1, color="#006699", label=label1)
                    axes[0].plot(tau, field2, color="#511D66", label=label2)

                    axes[1].semilogy(
                        list(range(modes.size)),
                        field1_k,
                        color="#006699",
                        label=label1_k,
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1].semilogy(
                        list(range(modes.size)),
                        field2_k,
                        color="#511D66",
                        label=label2_k,
                        marker="o",
                        markersize=3,
                        ls="None",
                    )

                    for ax in axes:

                        ax.grid(
                            True,
                            which="major",
                            axis="both",
                            color="gray",
                            ls="-",
                            lw=0.5,
                            alpha=0.2,
                        )
                        ax.legend(
                            loc="upper center",
                            fontsize=22,
                            ncol=2,
                            bbox_to_anchor=(0.5, 1.12),
                            frameon=False,
                        )

                        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

                        if ax is axes[0]:
                            ax.set_xlim(0, 3.5)
                            ax.set_xlabel(r"$\tau$", fontsize=30)
                            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

                        elif ax is axes[1]:
                            ax.set_xlim(0, 250)
                            ax.set_xlabel(r"$k$", fontsize=30)

                elif side == "right":

                    match field:

                        case "fc":
                            field1 = np.array(
                                result_dict["xcut12"]["Initial_Conditions"]["fc"]
                            ) - np.array(
                                result_dict["base"]["Initial_Conditions"]["fc"]
                            )
                            field2 = 2 ** (fac) * (
                                np.array(
                                    result_dict["xcut14"]["Initial_Conditions"]["fc"]
                                )
                                - np.array(
                                    result_dict["xcut12"]["Initial_Conditions"]["fc"]
                                )
                            )
                            field1_k = np.abs(
                                np.fft.rfft(
                                    result_dict["xcut12"]["Initial_Conditions"]["fc"]
                                )
                                - np.fft.rfft(
                                    result_dict["base"]["Initial_Conditions"]["fc"]
                                )
                            )[: int(tau.size // 4)]
                            field2_k = (
                                2 ** (fac)
                                * np.abs(
                                    np.fft.rfft(
                                        result_dict["xcut14"]["Initial_Conditions"][
                                            "fc"
                                        ]
                                    )
                                    - np.fft.rfft(
                                        result_dict["xcut12"]["Initial_Conditions"][
                                            "fc"
                                        ]
                                    )
                                )[: int(tau.size // 4)]
                            )
                            label1 = r"$f_c(2 \epsilon) - f_c(\epsilon)$"
                            label2 = (
                                f"$2^{{{fac}}}"
                                + r" \left[f_c(4 \epsilon) - f_c(2 \epsilon) \right]$"
                            )
                            label1_k = r"$\left|\hat{f}^k_c(2 \epsilon) - \hat{f}^k_c(\epsilon)\right|$"
                            label2_k = (
                                f"$2^{{{fac}}}"
                                + r" \left|\hat{f}^k_c(4 \epsilon) - \hat{f}^k_c(2 \epsilon) \right|$"
                            )
                        case "psic":
                            field1 = np.array(
                                result_dict["xcut12"]["Initial_Conditions"]["psic"]
                            ) - np.array(
                                result_dict["base"]["Initial_Conditions"]["psic"]
                            )
                            field2 = 2 ** (fac) * (
                                np.array(
                                    result_dict["xcut14"]["Initial_Conditions"]["psic"]
                                )
                                - np.array(
                                    result_dict["xcut12"]["Initial_Conditions"]["psic"]
                                )
                            )
                            field1_k = np.abs(
                                np.fft.rfft(
                                    result_dict["xcut12"]["Initial_Conditions"]["psic"]
                                )
                                - np.fft.rfft(
                                    result_dict["base"]["Initial_Conditions"]["psic"]
                                )
                            )[: int(tau.size // 4)]
                            field2_k = (
                                2 ** (fac)
                                * np.abs(
                                    np.fft.rfft(
                                        result_dict["xcut14"]["Initial_Conditions"][
                                            "psic"
                                        ]
                                    )
                                    - np.fft.rfft(
                                        result_dict["xcut12"]["Initial_Conditions"][
                                            "psic"
                                        ]
                                    )
                                )[: int(tau.size // 4)]
                            )
                            label1 = r"$\psi_c(2 \epsilon) - \psi_c(\epsilon)$"
                            label2 = (
                                f"$2^{{{fac}}}"
                                + r" \left[\psi_c(4 \epsilon) - \psi_c(2 \epsilon) \right]$"
                            )
                            label1_k = r"$\left|\hat{\psi}^k_c(2 \epsilon) - \hat{\psi}^k_c(\epsilon)\right|$"
                            label2_k = (
                                f"$2^{{{fac}}}"
                                + r" \left|\hat{\psi}^k_c(4 \epsilon) - \hat{\psi}^k_c(2 \epsilon) \right|$"
                            )
                        case "Up":
                            field1 = np.array(
                                result_dict["xcut12"]["Initial_Conditions"]["Up"]
                            ) - np.array(
                                result_dict["base"]["Initial_Conditions"]["Up"]
                            )
                            field2 = 2 ** (fac) * (
                                np.array(
                                    result_dict["xcut14"]["Initial_Conditions"]["Up"]
                                )
                                - np.array(
                                    result_dict["xcut12"]["Initial_Conditions"]["Up"]
                                )
                            )
                            field1_k = np.abs(
                                np.fft.rfft(
                                    result_dict["xcut12"]["Initial_Conditions"]["Up"]
                                )
                                - np.fft.rfft(
                                    result_dict["base"]["Initial_Conditions"]["Up"]
                                )
                            )[: int(tau.size // 4)]
                            field2_k = (
                                2 ** (fac)
                                * np.abs(
                                    np.fft.rfft(
                                        result_dict["xcut14"]["Initial_Conditions"][
                                            "Up"
                                        ]
                                    )
                                    - np.fft.rfft(
                                        result_dict["xcut12"]["Initial_Conditions"][
                                            "Up"
                                        ]
                                    )
                                )[: int(tau.size // 4)]
                            )
                            label1 = r"$U_p(2 \epsilon) - U_p(\epsilon)$"
                            label2 = (
                                f"$2^{{{fac}}}"
                                + r" \left[U_p(4 \epsilon) - U_p(2 \epsilon) \right]$"
                            )
                            label1_k = r"$\left|\hat{U}^k_p(2 \epsilon) - \hat{U}^k_p(\epsilon)\right|$"
                            label2_k = (
                                f"$2^{{{fac}}}"
                                + r" \left|\hat{U}^k_p(4 \epsilon) - \hat{U}^k_p(2 \epsilon) \right|$"
                            )

                    axes[0].plot(tau, field1, color="#006699", label=label1)
                    axes[0].plot(tau, field2, color="#511D66", label=label2)

                    axes[1].semilogy(
                        list(range(modes.size)),
                        field1_k,
                        color="#006699",
                        label=label1_k,
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1].semilogy(
                        list(range(modes.size)),
                        field2_k,
                        color="#511D66",
                        label=label2_k,
                        marker="o",
                        markersize=3,
                        ls="None",
                    )

                    for ax in axes:

                        ax.grid(
                            True,
                            which="major",
                            axis="both",
                            color="gray",
                            ls="-",
                            lw=0.5,
                            alpha=0.2,
                        )
                        ax.legend(
                            loc="upper center",
                            fontsize=22,
                            ncol=2,
                            bbox_to_anchor=(0.5, 1.12),
                            frameon=False,
                        )

                        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

                        if ax is axes[0]:
                            ax.set_xlim(0, 3.5)
                            ax.set_xlabel(r"$\tau$", fontsize=30)
                            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

                        elif ax is axes[1]:
                            ax.set_xlim(0, 250)
                            ax.set_xlabel(r"$k$", fontsize=30)

                plt.tight_layout()
                if save_fpath:
                    save_name = Path(
                        save_fpath.name.split(".")[0]
                        + f"_convergence_{side}_{field}"
                        + "."
                        + save_fpath.name.split(".")[-1]
                    )
                    plt.savefig((save_fpath.parent / save_name).as_posix())
                plt.show()

        else:

            for side in ["left", "right"]:

                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(32, 20))

                if side == "left":

                    fig.suptitle(f"Left Convergence for D={dim:.3f}", fontsize=25)
                    tau = np.array(result_dict["base"]["tau"])
                    fc1 = np.array(
                        result_dict["xcut02"]["Initial_Conditions"]["fc"]
                    ) - np.array(result_dict["base"]["Initial_Conditions"]["fc"])
                    fc2 = 2 ** (-6) * (
                        np.array(result_dict["xcut04"]["Initial_Conditions"]["fc"])
                        - np.array(result_dict["xcut02"]["Initial_Conditions"]["fc"])
                    )
                    psic1 = np.array(
                        result_dict["xcut02"]["Initial_Conditions"]["psic"]
                    ) - np.array(result_dict["base"]["Initial_Conditions"]["psic"])
                    psic2 = 2 ** (-6) * (
                        np.array(result_dict["xcut04"]["Initial_Conditions"]["psic"])
                        - np.array(result_dict["xcut02"]["Initial_Conditions"]["psic"])
                    )
                    Up1 = np.array(
                        result_dict["xcut02"]["Initial_Conditions"]["Up"]
                    ) - np.array(result_dict["base"]["Initial_Conditions"]["Up"])
                    Up2 = 2 ** (-6) * (
                        np.array(result_dict["xcut04"]["Initial_Conditions"]["Up"])
                        - np.array(result_dict["xcut02"]["Initial_Conditions"]["Up"])
                    )

                    modes = np.fft.rfftfreq(tau.size, d=tau[1] - tau[0])[
                        : int(tau.size // 4)
                    ]
                    fc1_k = np.abs(
                        np.fft.rfft(result_dict["xcut02"]["Initial_Conditions"]["fc"])
                        - np.fft.rfft(result_dict["base"]["Initial_Conditions"]["fc"])
                    )[: int(tau.size // 4)]
                    fc2_k = (
                        2 ** (-6)
                        * np.abs(
                            np.fft.rfft(
                                result_dict["xcut04"]["Initial_Conditions"]["fc"]
                            )
                            - np.fft.rfft(
                                result_dict["xcut02"]["Initial_Conditions"]["fc"]
                            )
                        )[: int(tau.size // 4)]
                    )
                    psic1_k = np.abs(
                        np.fft.rfft(result_dict["xcut02"]["Initial_Conditions"]["psic"])
                        - np.fft.rfft(result_dict["base"]["Initial_Conditions"]["psic"])
                    )[: int(tau.size // 4)]
                    psic2_k = (
                        2 ** (-6)
                        * np.abs(
                            np.fft.rfft(
                                result_dict["xcut04"]["Initial_Conditions"]["psic"]
                            )
                            - np.fft.rfft(
                                result_dict["xcut02"]["Initial_Conditions"]["psic"]
                            )
                        )[: int(tau.size // 4)]
                    )
                    Up1_k = np.abs(
                        np.fft.rfft(result_dict["xcut02"]["Initial_Conditions"]["Up"])
                        - np.fft.rfft(result_dict["base"]["Initial_Conditions"]["Up"])
                    )[: int(tau.size // 4)]
                    Up2_k = (
                        2 ** (-6)
                        * np.abs(
                            np.fft.rfft(
                                result_dict["xcut04"]["Initial_Conditions"]["Up"]
                            )
                            - np.fft.rfft(
                                result_dict["xcut02"]["Initial_Conditions"]["Up"]
                            )
                        )[: int(tau.size // 4)]
                    )

                    axes[0, 0].plot(
                        tau, fc1, color="#006699", label=r"$f_c(2x_L) - f_c(x_L)$"
                    )
                    axes[0, 0].plot(
                        tau,
                        fc2,
                        color="#511D66",
                        label=r"$2^{-6} \left(f_c(4x_L) - f_c(2x_L) \right)$",
                    )
                    axes[0, 1].plot(
                        tau,
                        psic1,
                        color="#006699",
                        label=r"$\psi_c(2x_L) - \psi_c(x_L)$",
                    )
                    axes[0, 1].plot(
                        tau,
                        psic2,
                        color="#511D66",
                        label=r"$2^{-6} \left(\psi_c(4x_L) - \psi_c(2x_L) \right)$",
                    )
                    axes[0, 2].plot(
                        tau, Up1, color="#006699", label=r"$U_p(2x_L) - U_p(x_L)$"
                    )
                    axes[0, 2].plot(
                        tau,
                        Up2,
                        color="#511D66",
                        label=r"$2^{-6} \left(U_p(4x_L) - U_p(2x_L) \right)$",
                    )

                    axes[1, 0].semilogy(
                        list(range(modes.size)),
                        fc1_k,
                        color="#006699",
                        label=r"$|\hat{f}^k_c(2x_L) - \hat{f}^k_c(x_L)|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 0].semilogy(
                        list(range(modes.size)),
                        fc2_k,
                        color="#511D66",
                        label=r"$2^{-6} \left|\hat{f}^k_c(4x_L) - \hat{f}^k_c(2x_L) \right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 1].semilogy(
                        list(range(modes.size)),
                        psic1_k,
                        color="#006699",
                        label=r"$|\hat{\psi}^k_c(2x_L) - \hat{\psi}^k_c(x_L)|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 1].semilogy(
                        list(range(modes.size)),
                        psic2_k,
                        color="#511D66",
                        label=r"$2^{-6} \left|\hat{\psi}^k_c(4x_L) - \hat{\psi}^k_c(2x_L) \right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 2].semilogy(
                        list(range(modes.size)),
                        Up1_k,
                        color="#006699",
                        label=r"$|\hat{U}^k_p(2x_L) - \hat{U}^k_p(x_L)|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 2].semilogy(
                        list(range(modes.size)),
                        Up2_k,
                        color="#511D66",
                        label=r"$2^{-6} \left|\hat{U}^k_p(4x_L) - \hat{U}^k_p(2x_L) \right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )

                    for axs, i in product(axes, range(3)):

                        axs[i].grid(
                            True,
                            which="major",
                            axis="both",
                            color="gray",
                            ls=":",
                            lw=0.5,
                        )
                        axs[i].legend(
                            loc="upper center",
                            fontsize=18,
                            ncol=2,
                            bbox_to_anchor=(0.5, 1.12),
                            frameon=False,
                        )

                        axs[i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

                        if axs[i] in axes[0]:
                            axs[i].set_xlim(0, 3.5)
                            axs[i].set_xlabel(r"$\tau$", fontsize=30)
                            axs[i].yaxis.set_minor_locator(
                                mpl.ticker.AutoMinorLocator(2)
                            )

                        elif axs[i] in axes[1]:
                            axs[i].set_xlim(0, 250)
                            axs[i].set_xlabel(r"$k$", fontsize=30)

                elif side == "right":

                    fig.suptitle(f"Right Convergence for D={dim:.3f}", fontsize=25)
                    tau = np.array(result_dict["base"]["tau"])
                    fc1 = np.array(
                        result_dict["xcut12"]["Initial_Conditions"]["fc"]
                    ) - np.array(result_dict["base"]["Initial_Conditions"]["fc"])
                    fc2 = 2 ** (fac) * (
                        np.array(result_dict["xcut14"]["Initial_Conditions"]["fc"])
                        - np.array(result_dict["xcut12"]["Initial_Conditions"]["fc"])
                    )
                    psic1 = np.array(
                        result_dict["xcut12"]["Initial_Conditions"]["psic"]
                    ) - np.array(result_dict["base"]["Initial_Conditions"]["psic"])
                    psic2 = 2 ** (fac) * (
                        np.array(result_dict["xcut14"]["Initial_Conditions"]["psic"])
                        - np.array(result_dict["xcut12"]["Initial_Conditions"]["psic"])
                    )
                    Up1 = result_dict["xcut12"]["Initial_Conditions"]["Up"] - np.array(
                        result_dict["base"]["Initial_Conditions"]["Up"]
                    )
                    Up2 = 2 ** (fac) * (
                        np.array(result_dict["xcut14"]["Initial_Conditions"]["Up"])
                        - np.array(result_dict["xcut12"]["Initial_Conditions"]["Up"])
                    )

                    modes = np.fft.rfftfreq(tau.size, d=tau[1] - tau[0])[
                        : int(tau.size // 4)
                    ]
                    fc1_k = np.abs(
                        np.fft.rfft(result_dict["xcut12"]["Initial_Conditions"]["fc"])
                        - np.fft.rfft(result_dict["base"]["Initial_Conditions"]["fc"])
                    )[: int(tau.size // 4)]
                    fc2_k = (
                        2 ** (fac)
                        * np.abs(
                            np.fft.rfft(
                                result_dict["xcut14"]["Initial_Conditions"]["fc"]
                            )
                            - np.fft.rfft(
                                result_dict["xcut12"]["Initial_Conditions"]["fc"]
                            )
                        )[: int(tau.size // 4)]
                    )
                    psic1_k = np.abs(
                        np.fft.rfft(result_dict["xcut12"]["Initial_Conditions"]["psic"])
                        - np.fft.rfft(result_dict["base"]["Initial_Conditions"]["psic"])
                    )[: int(tau.size // 4)]
                    psic2_k = (
                        2 ** (fac)
                        * np.abs(
                            np.fft.rfft(
                                result_dict["xcut14"]["Initial_Conditions"]["psic"]
                            )
                            - np.fft.rfft(
                                result_dict["xcut12"]["Initial_Conditions"]["psic"]
                            )
                        )[: int(tau.size // 4)]
                    )
                    Up1_k = np.abs(
                        np.fft.rfft(result_dict["xcut12"]["Initial_Conditions"]["Up"])
                        - np.fft.rfft(result_dict["base"]["Initial_Conditions"]["Up"])
                    )[: int(tau.size // 4)]
                    Up2_k = (
                        2 ** (fac)
                        * np.abs(
                            np.fft.rfft(
                                result_dict["xcut14"]["Initial_Conditions"]["Up"]
                            )
                            - np.fft.rfft(
                                result_dict["xcut12"]["Initial_Conditions"]["Up"]
                            )
                        )[: int(tau.size // 4)]
                    )

                    axes[0, 0].plot(
                        tau, fc1, color="#006699", label=r"$f_c(2x_R) - f_c(x_R)$"
                    )
                    axes[0, 0].plot(
                        tau,
                        fc2,
                        color="#511D66",
                        label=f"$2^{{{fac}}}"
                        + r" \left[f_c(4x_R) - f_c(2x_R) \right]$",
                    )
                    axes[0, 1].plot(
                        tau,
                        psic1,
                        color="#006699",
                        label=r"$\psi_c(2x_L) - \psi_c(x_R)$",
                    )
                    axes[0, 1].plot(
                        tau,
                        psic2,
                        color="#511D66",
                        label=f"$2^{{{fac}}}"
                        + r" \left[\psi_c(4x_R) - \psi_c(2x_R) \right]$",
                    )
                    axes[0, 2].plot(
                        tau, Up1, color="#006699", label=r"$U_p(2x_R) - U_p(x_R)$"
                    )
                    axes[0, 2].plot(
                        tau,
                        Up2,
                        color="#511D66",
                        label=f"$2^{{{fac}}}"
                        + r" \left[U_p(4x_R) - U_p(2x_R) \right]$",
                    )

                    axes[1, 0].semilogy(
                        list(range(modes.size)),
                        fc1_k,
                        color="#006699",
                        label=r"$\left|\hat{f}^k_c(2x_R) - \hat{f}^k_c(x_R)\right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 0].semilogy(
                        list(range(modes.size)),
                        fc2_k,
                        color="#242324",
                        label=f"$2^{{{fac}}}"
                        + r" \left|\hat{f}^k_c(4x_R) - \hat{f}^k_c(2x_R) \right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 1].semilogy(
                        list(range(modes.size)),
                        psic1_k,
                        color="#006699",
                        label=r"$\left|\hat{\psi}^k_c(2x_R) - \hat{\psi}^k_c(x_R)\right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 1].semilogy(
                        list(range(modes.size)),
                        psic2_k,
                        color="#511D66",
                        label=f"$2^{{{fac}}}"
                        + r" \left|\hat{\psi}^k_c(4x_R) - \hat{\psi}^k_c(2x_R) \right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 2].semilogy(
                        list(range(modes.size)),
                        Up1_k,
                        color="#006699",
                        label=r"$\left|\hat{U}^k_p(2x_R) - \hat{U}^k_p(x_R)\right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )
                    axes[1, 2].semilogy(
                        list(range(modes.size)),
                        Up2_k,
                        color="#511D66",
                        label=f"$2^{{{fac}}}"
                        + r" \left|\hat{U}^k_p(4x_R) - \hat{U}^k_p(2x_R) \right|$",
                        marker="o",
                        markersize=3,
                        ls="None",
                    )

                    for axs, i in product(axes, range(3)):

                        axs[i].grid(
                            True,
                            which="major",
                            axis="both",
                            color="gray",
                            ls=":",
                            lw=0.5,
                        )
                        axs[i].legend(
                            loc="upper center",
                            fontsize=18,
                            ncol=2,
                            bbox_to_anchor=(0.5, 1.12),
                            frameon=False,
                        )

                        axs[i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

                        if axs[i] in axes[0]:
                            axs[i].set_xlim(0, 3.5)
                            axs[i].set_xlabel(r"$\tau$", fontsize=30)
                            axs[i].yaxis.set_minor_locator(
                                mpl.ticker.AutoMinorLocator(2)
                            )

                        elif axs[i] in axes[1]:
                            axs[i].set_xlim(0, 250)
                            axs[i].set_xlabel(r"$k$", fontsize=30)

                plt.tight_layout()
                if save_fpath:
                    save_name = Path(
                        save_fpath.name.split(".")[0]
                        + f"_convergence_{side}"
                        + "."
                        + save_fpath.name.split(".")[-1]
                    )
                    plt.savefig((save_fpath.parent / save_name).as_posix())
                plt.show()

    def plot_fields(self, save_fpath: Path, single_plots: bool) -> None:

        result_dict = {}

        for file in self.input_files:
            _, dim, step = file.name.rsplit(".", maxsplit=1)[0].split("_")
            with open(file.as_posix(), "r") as f:
                result_dict[step] = json.load(f)

            result_dict[step]["x"] = np.array(
                sorted(list(result_dict[step].keys()))[:-1]
            )
            result_dict[step]["tau"] = np.linspace(
                0,
                result_dict[step]["Delta"],
                num=len(result_dict[step][str(result_dict[step]["x"][0])]["A"]),
            )

            As, Us, Vs, fs = [], [], [], []
            for x in result_dict[step]["x"]:
                x_str = str(x)
                As.append(result_dict[step][x_str]["A"])
                Us.append(result_dict[step][x_str]["U"])
                Vs.append(result_dict[step][x_str]["V"])
                fs.append(result_dict[step][x_str]["f"])
                del result_dict[step][x_str]

            result_dict[step]["a"] = np.array(As).T
            result_dict[step]["f"] = np.array(fs).T
            result_dict[step]["U"] = np.array(Us).T
            result_dict[step]["V"] = np.array(Vs).T

            result_dict[step]["x"], result_dict[step]["tau"] = np.meshgrid(
                np.array(result_dict[step]["x"], dtype=np.float64),
                result_dict[step]["tau"],
            )

            if single_plots:

                fields = ["a", "f", "U", "V"]
                cmaps = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Purples]
                axtitles = [
                    r"$a(x,\tau)$",
                    r"$f(x,\tau)$",
                    r"$U(x,\tau)$",
                    r"$V(x,\tau)$",
                ]

                for field, cmap, axtitle in zip(fields, cmaps, axtitles):

                    fig, ax = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(12, 10),
                        subplot_kw={"projection": "3d"},
                    )

                    x = result_dict[step]["x"]
                    t = result_dict[step]["tau"]
                    data = result_dict[step][field]

                    xmin, xmax = round_limits(x.min(), x.max())
                    ymin, ymax = round_limits(t.min(), t.max())
                    zmin, zmax = round_limits(data.min(), data.max())

                    if field == "a":
                        zmax = 1.5

                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode="soft")

                    ax.plot_surface(
                        x,
                        t,
                        data,
                        facecolors=rgb,
                        rstride=1,
                        cstride=1,
                        linewidth=0,
                        antialiased=True,
                        rasterized=True,
                    )

                    ax.zaxis.set_rotate_label(False)
                    ax.set_xlabel(r"$x$", fontsize=28, labelpad=20)
                    ax.set_ylabel(r"$\tau$", fontsize=28, labelpad=20)
                    ax.set_zlabel(f"{axtitle}", fontsize=28, labelpad=20, rotation=90)

                    ax.tick_params(axis="x", pad=10, labelsize=22)
                    ax.tick_params(axis="y", pad=10, labelsize=22)
                    ax.tick_params(axis="z", pad=10, labelsize=22, width=1.5, length=6)

                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
                    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
                    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

                    xticks = ax.get_xticks()
                    xtick_labels = [
                        f"{tick:.1f}" if tick == 0 else str(tick) for tick in xticks
                    ]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels)

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 3.5)
                    ax.set_zlim(zmin, zmax)

                    ax.view_init(elev=25, azim=-140)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False

                    plt.subplots_adjust(
                        left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
                    )
                    if save_fpath:
                        save_name = Path(
                            save_fpath.name.split(".")[0]
                            + f"_{field}_field_{dim:.3f}_{step}"
                            + "."
                            + save_fpath.name.split(".")[-1]
                        )
                        plt.savefig(
                            (save_fpath.parent / save_name).as_posix(),
                            bbox_inches="tight",
                        )
                    plt.show()

            else:

                fig, axes = plt.subplots(
                    nrows=2, ncols=2, figsize=(30, 25), subplot_kw={"projection": "3d"}
                )

                fields = ["a", "f", "U", "V"]
                cmaps = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Purples]
                axtitles = [
                    r"$a(x,\tau)$",
                    r"$f(x,\tau)$",
                    r"$U(x,\tau)$",
                    r"$V(x,\tau)$",
                ]

                for ax, field, cmap, axtitle in zip(
                    axes.flatten(), fields, cmaps, axtitles
                ):

                    x = result_dict[step]["x"]
                    t = result_dict[step]["tau"]
                    data = result_dict[step][field]

                    xmin, xmax = round_limits(x.min(), x.max())
                    ymin, ymax = round_limits(t.min(), t.max())
                    zmin, zmax = round_limits(data.min(), data.max())

                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode="soft")

                    ax.plot_surface(
                        x,
                        t,
                        data,
                        facecolors=rgb,
                        rstride=1,
                        cstride=1,
                        linewidth=0,
                        antialiased=True,
                        rasterized=True,
                    )

                    ax.zaxis.set_rotate_label(False)
                    ax.set_xlabel(r"$x$", fontsize=28, labelpad=20)
                    ax.set_ylabel(r"$\tau$", fontsize=28, labelpad=20)
                    ax.set_zlabel(f"{axtitle}", fontsize=28, labelpad=20, rotation=90)

                    ax.tick_params(axis="x", pad=10, labelsize=22)
                    ax.tick_params(axis="y", pad=10, labelsize=22)
                    ax.tick_params(axis="z", pad=10, labelsize=22, width=1.5, length=6)

                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
                    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
                    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

                    xticks = ax.get_xticks()
                    xtick_labels = [
                        f"{tick:.1f}" if tick == 0 else str(tick) for tick in xticks
                    ]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels)

                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    ax.set_zlim(zmin, zmax)

                    ax.view_init(elev=25, azim=-140)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False

                plt.subplots_adjust(
                    left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
                )

                if save_fpath:
                    save_name = Path(
                        save_fpath.name.split(".")[0]
                        + f"_fields_{dim:.3f}_{step}"
                        + "."
                        + save_fpath.name.split(".")[-1]
                    )
                    plt.savefig((save_fpath.parent / save_name).as_posix())

                plt.show()
    
    def plot_fields_paper(self, save_fpath: Path, single_plots: bool) -> None:

        result_dict = {}

        for file in self.input_files:
            _, dim, step = file.name.rsplit(".", maxsplit=1)[0].split("_")
            with open(file.as_posix(), "r") as f:
                result_dict[step] = json.load(f)

            result_dict[step]["x"] = np.array(
                sorted(list(result_dict[step].keys()))[:-1]
            )
            result_dict[step]["tau"] = np.linspace(
                0,
                result_dict[step]["Delta"],
                num=len(result_dict[step][str(result_dict[step]["x"][0])]["A"]),
            )

            As, Us, Vs, fs = [], [], [], []
            for x in result_dict[step]["x"]:
                x_str = str(x)
                As.append(result_dict[step][x_str]["A"])
                Us.append(result_dict[step][x_str]["U"])
                Vs.append(result_dict[step][x_str]["V"])
                fs.append(result_dict[step][x_str]["f"])
                del result_dict[step][x_str]

            result_dict[step]["a"] = np.array(As).T
            result_dict[step]["f"] = np.array(fs).T
            result_dict[step]["U"] = np.array(Us).T
            result_dict[step]["V"] = np.array(Vs).T

            result_dict[step]["x"], result_dict[step]["tau"] = np.meshgrid(
                np.array(result_dict[step]["x"], dtype=np.float64),
                result_dict[step]["tau"],
            )

            ia2 = 1 / result_dict[step]["a"]**(2)
            U = result_dict[step]["U"]
            V = result_dict[step]["V"]
            x = result_dict[step]["x"]
            t = result_dict[step]["tau"]
            delta = result_dict[step]["Delta"]

            result_dict[step]["omega"] = -np.log(ia2)
            result_dict[step]["pi"] = (V+U) / (2*x) * np.sqrt((float(dim) - 2)**3 / 4)
            result_dict[step]["psi"] = (V-U) / (2*x**2) * np.sqrt((float(dim) - 2)**3 / 4)

            if single_plots:

                fields = ["omega", "f", "pi", "psi"]
                cmaps = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Purples]
                axtitles = [
                    r"$\omega(x,\tau)$",
                    r"$f(x,\tau)$",
                    r"$\pi(x,\tau)$",
                    r"$\psi(x,\tau)$",
                ]

                for field, cmap, axtitle in zip(fields, cmaps, axtitles):

                    fig, ax = plt.subplots(
                        nrows=1,
                        ncols=1,
                        figsize=(12, 10),
                        subplot_kw={"projection": "3d"},
                    )

                    data = result_dict[step][field]

                    xmin, xmax = round_limits(x.min(), x.max())
                    ymin, ymax = round_limits(t.min(), t.max())
                    zmin, zmax = round_limits(data.min(), data.max())

                    if field == "a":
                        zmax = 1.5

                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode="soft")

                    ax.plot_surface(
                        x,
                        t/delta,
                        data,
                        facecolors=rgb,
                        rstride=1,
                        cstride=1,
                        linewidth=0,
                        antialiased=True,
                        rasterized=True,
                    )

                    ax.zaxis.set_rotate_label(False)
                    ax.set_xlabel(r"$x$", fontsize=28, labelpad=20)
                    ax.set_ylabel(r"$\tau / \Delta$", fontsize=28, labelpad=20)
                    ax.set_zlabel(f"{axtitle}", fontsize=28, labelpad=20, rotation=90)

                    ax.tick_params(axis="x", pad=10, labelsize=22)
                    ax.tick_params(axis="y", pad=10, labelsize=22)
                    ax.tick_params(axis="z", pad=10, labelsize=22, width=1.5, length=6)

                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
                    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
                    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

                    xticks = ax.get_xticks()
                    xtick_labels = [
                        f"{tick:.1f}" if tick == 0 else str(tick) for tick in xticks
                    ]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels)
                    ax.set_yticks(xticks)
                    ax.set_yticklabels(xtick_labels)

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_zlim(zmin, zmax)

                    ax.view_init(elev=25, azim=-140)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False

                    plt.subplots_adjust(
                        left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
                    )
                    if save_fpath:
                        save_name = Path(
                            save_fpath.name.split(".")[0]
                            + f"_{field}_field_{float(dim):.3f}_{step}_paper"
                            + "."
                            + save_fpath.name.split(".")[-1]
                        )
                        plt.savefig(
                            (save_fpath.parent / save_name).as_posix(),
                            bbox_inches="tight",
                        )
                    plt.show()

            else:

                fig, axes = plt.subplots(
                    nrows=2, ncols=2, figsize=(30, 25), subplot_kw={"projection": "3d"}
                )

                fields = ["omega", "f", "pi", "psi"]
                cmaps = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Purples]
                axtitles = [
                    r"$\omega(x,\tau)$",
                    r"$f(x,\tau)$",
                    r"$\pi(x,\tau)$",
                    r"$\psi(x,\tau)$",
                ]

                for ax, field, cmap, axtitle in zip(
                    axes.flatten(), fields, cmaps, axtitles
                ):

                    x = result_dict[step]["x"]
                    t = result_dict[step]["tau"]
                    data = result_dict[step][field]

                    xmin, xmax = round_limits(x.min(), x.max())
                    ymin, ymax = round_limits(t.min(), t.max())
                    zmin, zmax = round_limits(data.min(), data.max())

                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode="soft")

                    ax.plot_surface(
                        x,
                        t/delta,
                        data,
                        facecolors=rgb,
                        rstride=1,
                        cstride=1,
                        linewidth=0,
                        antialiased=True,
                        rasterized=True,
                    )

                    ax.zaxis.set_rotate_label(False)
                    ax.set_xlabel(r"$x$", fontsize=28, labelpad=20)
                    ax.set_ylabel(r"$\tau / \Delta$", fontsize=28, labelpad=20)
                    ax.set_zlabel(f"{axtitle}", fontsize=28, labelpad=20, rotation=90)

                    ax.tick_params(axis="x", pad=10, labelsize=22)
                    ax.tick_params(axis="y", pad=10, labelsize=22)
                    ax.tick_params(axis="z", pad=10, labelsize=22, width=1.5, length=6)

                    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
                    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
                    ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

                    xticks = ax.get_xticks()
                    xtick_labels = [
                        f"{tick:.1f}" if tick == 0 else str(tick) for tick in xticks
                    ]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels)
                    ax.set_yticks(xticks)
                    ax.set_yticklabels(xtick_labels)

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_zlim(zmin, zmax)

                    ax.view_init(elev=25, azim=-140)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False

                plt.subplots_adjust(
                    left=0, right=1, top=1, bottom=0, wspace=0, hspace=0
                )

                if save_fpath:
                    save_name = Path(
                        save_fpath.name.split(".")[0]
                        + f"_fields_{float(dim):.3f}_{step}_paper"
                        + "."
                        + save_fpath.name.split(".")[-1]
                    )
                    plt.savefig((save_fpath.parent / save_name).as_posix())

                plt.show()

    def plot_mismatch_pos_finder(self, save_fpath: Path, spec: str = None) -> None:

        result_dict = {}

        for file in self.input_files:
            _, dim, step = file.name.rsplit(".", maxsplit=1)[0].split("_")
            with open(file.as_posix(), "r") as f:
                result_dict[step] = json.load(f)

            result_dict[step]["x"] = np.array(
                sorted(list(result_dict[step].keys()))[:-1]
            )
            result_dict[step]["tau"] = np.linspace(
                0,
                result_dict[step]["Delta"],
                num=len(result_dict[step][str(result_dict[step]["x"][0])]["A"]),
            )

            As, Us, Vs, fs = [], [], [], []
            for x in result_dict[step]["x"]:
                x_str = str(x)
                As.append(result_dict[step][x_str]["A"])
                Us.append(result_dict[step][x_str]["U"])
                Vs.append(result_dict[step][x_str]["V"])
                fs.append(result_dict[step][x_str]["f"])
                del result_dict[step][x_str]

            result_dict[step]["a"] = np.array(As).T
            result_dict[step]["f"] = np.array(fs).T
            result_dict[step]["U"] = np.array(Us).T
            result_dict[step]["V"] = np.array(Vs).T

            result_dict[step]["x"] = np.array(result_dict[step]["x"], dtype=np.float64)
            if spec:
                result_dict[step]["psi"] = (
                    result_dict[step]["V"] - result_dict[step]["U"]
                ) / (2 * result_dict[step]["x"] ** 2)
                label = r"$2x^2 \Psi=V-U$"
            else:
                result_dict[step]["psi"] = (
                    result_dict[step]["V"] - result_dict[step]["U"]
                )
                label = r"$\Psi=\frac{V-U}{2x^2}$"

            result_dict[step]["psi"] = np.max(result_dict[step]["psi"], axis=0)

            arg_max_psi = np.argmax(result_dict[step]["psi"])
            x_max_psi = result_dict[step]["x"][arg_max_psi]

            for field in ["a", "f", "U", "V"]:
                del result_dict[step][field]

            fig, ax = plt.subplots(figsize=(12, 8))

            ax.plot(
                result_dict[step]["x"],
                result_dict[step]["psi"],
                label=label,
                ls="-",
                color="#006699",
            )

            ax.axvline(
                x_max_psi,
                label=r"$\max \limits_{x} \Psi=$" + f"{x_max_psi}",
                color="#8C1B3D",
                ls="--",
            )

            ymin, ymax = round_limits(
                result_dict[step]["psi"].min(), result_dict[step]["psi"].max()
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(r"$x$", fontsize=28, labelpad=10)
            ax.set_ylabel(r"$\Psi$", fontsize=28, labelpad=10)

            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

            ax.grid(
                color="grey", which="major", linestyle="-", linewidth=0.5, alpha=0.2
            )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncols=2,
                frameon=False,
                fontsize=20,
            )

            plt.tight_layout()

        if save_fpath:
            save_name = Path(
                save_fpath.name.split(".")[0]
                + f"_mismatch_finder_{dim}_{step}"
                + "."
                + save_fpath.name.split(".")[-1]
            )
            plt.savefig((save_fpath.parent / save_name).as_posix())

        plt.show()

    def plot_initial_data(
        self, save_fpath: Path, dim: str, experimental_data: str, spec: str
    ) -> None:

        fpath = Path(self.input_files[0])
        with open(fpath.as_posix(), "r") as f:
            result_dict = json.load(f)

        if dim:
            dim = list(
                filter(
                    lambda x: np.isclose(float(x), float(dim)), list(result_dict.keys())
                )
            )[0]
            result_dict = result_dict[dim]
        else:
            dim = result_dict["Dim"]

        assert result_dict["Converged"], "Data has to be converged to be displayed!"

        echoing_period = result_dict["Initial_Conditions"]["Delta"]
        t = np.linspace(0, echoing_period, num=result_dict["Ntau"])
        fc = np.array(result_dict["Initial_Conditions"]["fc"], dtype=np.float64)
        psic = np.array(result_dict["Initial_Conditions"]["psic"], dtype=np.float64)
        Up = np.array(result_dict["Initial_Conditions"]["Up"], dtype=np.float64)

        if not spec:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(t, fc, label=r"$f_c$", ls="-", color="#006699")
            ax.plot(t, Up, label=r"$U_p$", ls="-.", color="#006699")
            ax2 = ax.twinx()
            ax2.plot(t, psic, label=r"$\psi_c$", ls="--", color="#511D66")

            lines_d, labels_d = ax.get_legend_handles_labels()
            lines2_d, labels2_d = ax2.get_legend_handles_labels()

            for label in ax.get_yticklabels():
                label.set_color("#006699")

            for label in ax2.get_yticklabels():
                label.set_color("#511D66")

        if experimental_data:

            with open(experimental_data, "r") as f:
                result_dict_e = json.load(f)

            name = list(result_dict_e.keys())[0]
            result_dict_e = result_dict_e[name]

            echoing_period_e = result_dict_e["Delta"]
            t_e = np.linspace(
                0, echoing_period_e, num=len(result_dict_e["fc"]), dtype=np.float64
            )
            fc_e = np.array(result_dict_e["fc"], dtype=np.float64)
            psic_e = np.array(result_dict_e["psic"], dtype=np.float64)
            Up_e = np.array(result_dict_e["Up"], dtype=np.float64)

            if spec:

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_yscale("log")
                ax.plot(
                    t,
                    np.abs(fc - fc_e),
                    label=r"$|\Delta f_c|$",
                    ls="--",
                    color="#006699",
                )
                ax.plot(
                    t,
                    np.abs(Up - Up_e),
                    label=r"$|\Delta U_p|$",
                    ls="-",
                    color="#A60808",
                )
                ax.plot(
                    t,
                    np.abs(psic - psic_e),
                    label=r"$|\Delta \psi_c|$",
                    ls="-",
                    color="#E0880D",
                )

                ax.set_xlim(0, np.ceil(t.max() * 10) / 10)
                ax.set_ylim(1e-16, 1e-6)

                ax.set_xlabel(r"$\tau$", fontsize=28)
                ax.set_ylabel(r"Initial Data Deviations", fontsize=28)

                ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

            else:
                ax.plot(
                    t_e, fc_e, label=r"$f_c$" + " (FORTRAN)", ls="--", color="#A60808"
                )
                ax.plot(
                    t_e,
                    Up_e,
                    label=r"$U_p$" + " (FORTRAN)",
                    ls="-",
                    color="#E377C2",
                    zorder=0,
                )
                ax2.plot(
                    t_e,
                    psic_e,
                    label=r"$\psi_c$" + " (FORTRAN)",
                    ls="-",
                    color="#E0880D",
                    zorder=1,
                )

                lines_e, labels_e = ax.get_legend_handles_labels()
                lines2_e, labels2_e = ax2.get_legend_handles_labels()

                lines_e = list(set(lines_e).difference(set(lines_d)))
                labels_e = list(set(labels_e).difference(set(labels_d)))
                lines2_e = list(set(lines2_e).difference(set(lines2_d)))
                labels2_e = list(set(labels2_e).difference(set(labels2_d)))

                fcmin, fcmax = round_limits(fc.min(), fc.max())
                Upmin, Upmax = round_limits(Up.min(), Up.max())
                fcmin = fcmin if fcmin < Upmin else Upmin
                fcmax = fcmax if fcmax > Upmax else Upmax
                psicmin, psicmax = round_limits(psic.min(), psic.max())

                ax.set_ylim(fcmin, fcmax)
                ax2.set_ylim(psicmin, psicmax)

                ax.set_xlabel(r"$\tau$", fontsize=28)

                ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
                ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

            print(f"Max deviation fc: {np.max(np.abs(fc_e - fc)):.4g}")
            print(f"Max deviation psic: {np.max(np.abs(psic_e - psic)):.4g}")
            print(f"Max deviation Up: {np.max(np.abs(Up_e - Up)):.4g}")
            print(f"Deviation Delta: {np.abs(echoing_period_e - echoing_period):.4g}")

        ax.grid(color="grey", which="major", linestyle="-", linewidth=0.5, alpha=0.2)
        ax.set_xlim(0, np.ceil(t.max() * 10) / 10)

        if spec:

            ax.legend(
                loc="upper center",
                fontsize=18,
                ncol=3,
                bbox_to_anchor=(0.5, 1.1),
                frameon=False,
            )

        else:

            if experimental_data:

                ax.legend(
                    [lines_d[0]]
                    + [lines_e[0]]
                    + [lines_d[1]]
                    + [lines_e[1]]
                    + [lines2_d[0]]
                    + [lines2_e[0]],
                    [labels_d[0]]
                    + [labels_e[0]]
                    + [labels_d[1]]
                    + [labels_e[1]]
                    + [labels2_d[0]]
                    + [labels2_e[0]],
                    loc="upper center",
                    fontsize=18,
                    ncol=3,
                    bbox_to_anchor=(0.5, 1.18),
                    frameon=False,
                )
            else:

                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()

                ax.legend(
                    lines + lines2,
                    labels + labels2,
                    loc="upper center",
                    fontsize=20,
                    ncol=3,
                    bbox_to_anchor=(0.5, 1.10),
                    frameon=False,
                )

        plt.tight_layout()

        if save_fpath:
            if experimental_data:
                save_name = Path(
                    save_fpath.name.split(".")[0]
                    + f"_initial-data_{dim}_fortran"
                    + "."
                    + save_fpath.name.split(".")[-1]
                )
            elif experimental_data and spec:
                save_name = Path(
                    save_fpath.name.split(".")[0]
                    + f"_initial-data_{dim}_differences"
                    + "."
                    + save_fpath.name.split(".")[-1]
                )
            else:
                save_name = Path(
                    save_fpath.name.split(".")[0]
                    + f"_initial-data_{dim}"
                    + "."
                    + save_fpath.name.split(".")[-1]
                )
            plt.savefig((save_fpath.parent / save_name).as_posix())

        plt.show()

    def plot_echoing_period(
        self, save_fpath: Path, experimental_data: str, spec: str
    ) -> None:

        input_dict = result_dict = {}

        result_dict["dims"] = []
        result_dict["periods"] = []

        if len(self.input_files) > 1:
            fpaths = [Path(file) for file in self.input_files]
            for fpath in fpaths:
                with open(fpath.as_posix(), "r") as f:
                    tmp_dict = json.load(f)
                input_dict.update(tmp_dict)

            for key in input_dict.keys():
                if input_dict[key]["Converged"] == True:
                    result_dict["dims"].append(input_dict[key]["Dim"])
                    result_dict["periods"].append(
                        input_dict[key]["Initial_Conditions"]["Delta"]
                    )

        else:
            fpath = Path(self.input_files[0])

            with open(fpath.as_posix(), "r") as f:
                input_dict = json.load(f)

            for key in input_dict.keys():
                if input_dict[key]["Converged"] == True:
                    result_dict["dims"].append(input_dict[key]["Dim"])
                    result_dict["periods"].append(
                        input_dict[key]["Initial_Conditions"]["Delta"]
                    )

        result_dict["dims"] = np.array(result_dict["dims"], dtype=np.float64)
        result_dict["periods"] = np.array(result_dict["periods"], dtype=np.float64)
        sorted_indices = np.argsort(result_dict["dims"])

        dims = result_dict["dims"][sorted_indices]
        periods = result_dict["periods"][sorted_indices]

        period_max_arg = np.argmax(periods)
        period_max = periods[period_max_arg]
        dim_max = dims[period_max_arg]

        fig, ax = plt.subplots(figsize=(12, 8))

        if not spec:

            ax.plot(dims, periods, label="C++ code", ls="-", color="#006699")
            ax.plot(
                dim_max,
                period_max,
                label=r"$\Delta_{max}=$" + f"{period_max:.5f}",
                ls="",
                marker="*",
                markersize=15,
                color="#006699",
            )

        if experimental_data:

            with open(experimental_data, "r") as f:
                result_dict_e = json.load(f)

            name_e = list(result_dict_e.keys())[0]
            result_dict_e = result_dict_e[name_e]
            dims_e = result_dict_e["Dims"]
            periods_e = result_dict_e["Periods"]
            period_max_arg_e = np.argmax(periods_e)
            period_max_e = periods_e[period_max_arg_e]
            dim_max_e = dims_e[period_max_arg_e]

            dims_e_linspace = np.arange(3.25, 4.005, 0.005)
            periods_e_spline = CubicSpline(dims_e, periods_e, bc_type="natural")

            filtered_indices = [i for i, item in enumerate(dims) if item in dims_e]
            max_deviation_period = np.max(
                np.abs(np.array(periods)[filtered_indices] - np.array(periods_e))
            )
            max_deviation_index = np.argmax(
                np.abs(np.array(periods)[filtered_indices] - np.array(periods_e))
            )
            print(
                f"Max echoing period deviaton: {max_deviation_period:.5g} at D={np.array(dims)[filtered_indices][max_deviation_index]}"
            )

            if spec:
                ax.plot(
                    dims_e,
                    np.abs(np.array(periods)[filtered_indices] - np.array(periods_e)),
                    label=f"Echoing Period Difference",
                    ls="--",
                    color="#A60808",
                    marker="*",
                    markersize=10,
                )
                ax.set_xlim(dims.min() * 0.99, dims.max() * 1.01)
                ax.set_yscale("log")
                ax.set_ylim(1e-12, 1e-2)
                ax.set_xlabel(r"Dimension D", fontsize=28, labelpad=10)
                ax.set_ylabel(
                    r"$\left| \Delta_{\mathrm{\scriptsize C++}} - \Delta_{\mathrm{\scriptsize FORTRAN}} \right|$",
                    fontsize=28,
                    labelpad=10,
                )
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
                ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
                ax.grid(
                    color="grey", which="major", linestyle="-", linewidth=0.5, alpha=0.2
                )
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.1),
                    ncols=2,
                    frameon=False,
                    fontsize=20,
                )
            else:
                ax.plot(
                    dims_e_linspace,
                    periods_e_spline(dims_e_linspace),
                    label=f"{name_e} code",
                    ls="--",
                    color="#A60808",
                )
                ax.plot(
                    dim_max_e,
                    period_max_e,
                    label=r"$\Delta_{max}=$" + f"{period_max:.5f}",
                    ls="",
                    marker="*",
                    markersize=8,
                    color="#A60808",
                )
                ax.set_xlim(dims.min() * 0.99, dims.max() * 1.01)
                ax.set_ylim(3.1, 3.5)
                ax.set_xlabel(r"Dimension D", fontsize=28, labelpad=10)
                ax.set_ylabel(r"Echoing Period $\Delta$", fontsize=28, labelpad=10)
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
                ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
                ax.grid(
                    color="grey", which="major", linestyle="-", linewidth=0.5, alpha=0.2
                )
                if experimental_data:
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1.16),
                        ncols=2,
                        frameon=False,
                        fontsize=20,
                    )
                else:
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1.1),
                        ncols=2,
                        frameon=False,
                        fontsize=20,
                    )

        plt.tight_layout()

        if save_fpath:
            if experimental_data:
                save_name = Path(
                    save_fpath.name.split(".")[0]
                    + "_echoing-period_experimental"
                    + "."
                    + save_fpath.name.split(".")[-1]
                )
            elif experimental_data and spec:
                save_name = Path(
                    save_fpath.name.split(".")[0]
                    + "_echoing-period_experimental_differences"
                    + "."
                    + save_fpath.name.split(".")[-1]
                )
            else:
                save_name = Path(
                    save_fpath.name.split(".")[0]
                    + "_echoing-period"
                    + "."
                    + save_fpath.name.split(".")[-1]
                )

            plt.savefig((save_fpath.parent / save_name).as_posix())

        plt.show()

    def plot_max_speedup(self, save_fpath: Path, dim: str | float) -> None:

        assert (
            len(self.input_files) == 1
        ), "Only the path folder should be given for benchmark plots!"

        fpath = Path(self.input_files[0]).absolute()

        assert fpath.is_dir(), "The input path should be a folder, not a file!"

        benchmark_files = glob((fpath / "benchmark").as_posix() + "*/*", recursive=True)
        benchmark_files = [
            Path(file)
            for file in benchmark_files
            if (f"{float(dim):.6f}" in file and "Serial" in file)
        ]

        result_serial = {}

        cores = np.logspace(0, 17, base=2, num=100)
        amdahl_law = lambda c, f: 1 / ((1 - f) + f / c)

        fig, ax = plt.subplots(figsize=(12, 8))
        markers = ["o", "s", "^", "D", "v", ">", "<", "P", "*"]
        colors = [
            "navy",
            "goldenrod",
            "darkviolet",
            "teal",
            "crimson",
            "darkorange",
            "forestgreen",
            "indigo",
            "maroon",
        ]

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(1, 2**17)
        ax.set_ylim(1, 1e3)
        ax.set_xlabel("Cores [-]", fontsize=28, labelpad=10)
        ax.set_ylabel("Speedup [-]", fontsize=28, labelpad=10)
        ax.set_xticks([2**i for i in range(18) if i % 2 == 0])
        ax.set_xticklabels([f"$2^{{{i}}}$" for i in range(18) if i % 2 == 0])

        for i, file in enumerate(benchmark_files):

            with open(file, "r") as f:
                tmp_dict = json.load(f)
            mean_time = 0
            repetitions = tmp_dict["Repetitions"]
            Ntau = tmp_dict["Ntau"]
            Nleft = tmp_dict["NLeft"]
            Nright = tmp_dict["NRight"]

            newton_keys = [key for key in tmp_dict["0"].keys() if "NewtonStep" in key]
            assemble_keys = [
                key for key in tmp_dict["0"].keys() if "AssembleStep" in key
            ]
            mean_time = np.mean(
                [tmp_dict[f"{i}"]["OverallTime"] for i in range(repetitions)]
            )
            mean_newton_time = {key: 0 for key in newton_keys}
            mean_assemble_time = {key: 0 for key in assemble_keys}
            for j, (nkey, akey) in product(
                range(repetitions), zip(newton_keys, assemble_keys)
            ):
                mean_newton_time[nkey] += tmp_dict[f"{j}"][nkey] / repetitions
                mean_assemble_time[akey] += tmp_dict[f"{j}"][akey] / repetitions

            overall_newton_time = np.sum([n for n in mean_newton_time.values()])
            overall_assemble_time = np.sum([a for a in mean_assemble_time.values()])

            par_frac = overall_assemble_time / mean_time
            theo_su = 1 / ((1 - par_frac))
            if Ntau in result_serial.keys():
                if theo_su > result_serial[Ntau]["val"]:
                    result_serial[Ntau] = {"val": theo_su, "color": colors[i]}
            else:
                result_serial[Ntau] = {"val": theo_su, "color": colors[i]}

            print(
                f"For D={float(dim):.3f} with N_tau={Ntau}, N_left={Nleft}, "
                + f"N_right={Nright}, the mean execution time is {mean_time:.5g} [s] and mean "
                + f"mean assemble time is {overall_assemble_time:.5g}, which results in a theoretical "
                + f"parallel fraction of {par_frac:.6f} with maximum speed up of {theo_su:.2f} for (p->oo)."
            )

            label = r"$N_\tau= $" + f"${Ntau}, \\, N_L=N_R=$ " + f"${Nleft}$"
            ax.plot(cores, amdahl_law(cores, par_frac), label=label, color=colors[i])

        for key in result_serial:
            ax.axhline(
                result_serial[key]["val"], ls="--", color=result_serial[key]["color"]
            )
            ax.text(
                1.5,
                result_serial[key]["val"] * 1.1,
                f"$SU_{{max}} \\approx{result_serial[key]['val']:.0f}$",
                color=result_serial[key]["color"],
                fontsize=20,
            )

        lines, labels = ax.get_legend_handles_labels()

        lines = [lines[5], lines[3], lines[1], lines[2], lines[4], lines[0]]
        labels = [labels[5], labels[3], labels[1], labels[2], labels[4], labels[0]]

        ax.grid(color="grey", which="major", linestyle="-", linewidth=0.5, alpha=0.2)

        ax.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncols=3,
            frameon=False,
            fontsize=15,
        )

        plt.tight_layout()

        if save_fpath:
            save_name = Path(
                save_fpath.name.split(".")[0]
                + "_theoretical_speedup"
                + "."
                + save_fpath.name.split(".")[-1]
            )
            fig.savefig((save_fpath.parent / save_name).as_posix())

        plt.show()

    def plot_benchmark(
        self, save_fpath: Path, dim: str | float, rel_su: bool = False
    ) -> None:

        assert (
            len(self.input_files) == 1
        ), "Only the path folder should be given for benchmark plots!"

        fpath = Path(self.input_files[0]).absolute()

        assert fpath.is_dir(), "The input path should be a folder, not a file!"

        benchmark_files = glob((fpath / "benchmark").as_posix() + "*")
        benchmark_files = [Path(file) for file in benchmark_files if dim in file]

        result_omp = {}
        result_mpi = {}
        result_hybrid = {}
        result_serial = {}

        Ntau = Nleft = Nright = None

        for file in benchmark_files:
            kname = file.name.rsplit(".", maxsplit=1)[0].split("_")[2]
            with open(file, "r") as f:
                tmp_dict = json.load(f)
            mean_time = 0
            repetitions = tmp_dict["Repetitions"]
            Ntau = tmp_dict["Ntau"]
            Nleft = tmp_dict["NLeft"]
            Nright = tmp_dict["NRight"]

            newton_keys = [key for key in tmp_dict["0"].keys() if "NewtonStep" in key]
            assemble_keys = [
                key for key in tmp_dict["0"].keys() if "AssembleStep" in key
            ]
            mean_time = np.mean(
                [tmp_dict[f"{i}"]["OverallTime"] for i in range(repetitions)]
            )
            mean_newton_time = {key: 0 for key in newton_keys}
            mean_assemble_time = {key: 0 for key in assemble_keys}
            for j, (nkey, akey) in product(
                range(repetitions), zip(newton_keys, assemble_keys)
            ):
                mean_newton_time[nkey] += tmp_dict[f"{j}"][nkey] / repetitions
                mean_assemble_time[akey] += tmp_dict[f"{j}"][akey] / repetitions

            overall_newton_time = np.sum([n for n in mean_newton_time.values()])
            overall_assemble_time = np.sum([a for a in mean_assemble_time.values()])

            match kname:
                case "OpenMP":
                    threads = tmp_dict["Threads"]
                    result_omp[threads] = mean_time
                case "MPI":
                    cores = tmp_dict["Cores"]
                    result_mpi[cores] = mean_time
                case "Hybrid":
                    cores = tmp_dict["Cores"]
                    threads = tmp_dict["Threads"]
                    result_hybrid[int(cores * threads)] = mean_time
                case "Serial":
                    result_serial["par_frac"] = overall_assemble_time / mean_time
                    result_serial["t"] = mean_time
                case "_":
                    raise ValueError(
                        f"{kname} is not defined in the evaluation algorithm!"
                    )

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Cores [-]", fontsize=28, labelpad=10)
        if rel_su:
            ax.set_ylabel("Rel. Speedup [-]", fontsize=28, labelpad=10)
        else:
            ax.set_ylabel("Speedup [-]", fontsize=28, labelpad=10)
        max_speed = 0
        max_procs = 0
        num = 0
        theo_su_max = 1 / (1 - result_serial["par_frac"])

        markers = ["o", "s", "^", "D", "v", ">", "<", "P", "*"]
        colors = ["#0F52BA", "#9B111E", "#0E7C61"]

        for i, (dic, kind) in enumerate(
            zip([result_omp, result_mpi, result_hybrid], ["OpenMP", "MPI", "Hybrid"])
        ):

            if dic:
                num += 1
                procs = np.array([key for key in dic.keys()])
                args_procs = np.argsort(procs)
                procs = procs[args_procs]
                max_procs = procs.max() if procs.max() > max_procs else max_procs
                overall_times = np.array([val for val in dic.values()])[args_procs]

                if rel_su:
                    overall_times = (1 / overall_times[0] * overall_times) ** (-1)
                else:
                    overall_times = (1 / result_serial["t"] * overall_times) ** (-1)
                    overall_times[0] = 1.0
                max_speed = (
                    overall_times.max()
                    if overall_times.max() > max_speed
                    else max_speed
                )

                ax.plot(
                    procs,
                    overall_times,
                    label=kind,
                    ls="-",
                    marker=markers[i],
                    color=colors[i],
                )

                print(
                    f"{kind}, {Ntau=}, Nleft={Nright=}, #cores: {procs}, SUs: {overall_times}"
                )

        cores_max_exp = int(np.ceil(np.log2(max_procs)))
        cores = np.logspace(0, cores_max_exp, base=2, num=100)
        amdahl_law = lambda c, f: 1 / ((1 - f) + f / c)
        theo_su = amdahl_law(cores, result_serial["par_frac"])

        ax.plot(
            cores,
            theo_su,
            ls="--",
            label=f"Amdahl (f={result_serial['par_frac']:.4f})",
            color="#1A1A1A",
        )
        ax.axhline(theo_su_max, ls="--", color="#1A1A1A")
        ax.text(
            1.5,
            theo_su_max * 1.1,
            f"$SU_{{max}} \\approx{theo_su_max:.0f}$",
            color="#1A1A1A",
            fontsize=20,
        )

        ax.set_xlim(1, 2**cores_max_exp)
        ax.set_ylim(1, 1e3)
        ax.set_xticks([2**i for i in range(cores_max_exp + 1) if i % 2 == 0])
        ax.set_xticklabels(
            [f"$2^{{{i}}}$" for i in range(cores_max_exp + 1) if i % 2 == 0]
        )

        ax.grid(color="grey", which="major", linestyle="-", linewidth=0.5, alpha=0.2)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncols=num + 1,
            frameon=False,
            fontsize=20,
        )

        plt.tight_layout()

        if save_fpath:
            save_name = Path(
                save_fpath.name.split(".")[0]
                + f"_benchmark_{Ntau}_{Nleft}"
                + "."
                + save_fpath.name.split(".")[-1]
            )
            fig.savefig((save_fpath.parent / save_name).as_posix())

        plt.show()

    def plot_efficiency(
        self, save_fpath: Path, dim: str | float, rel_su: bool = False
    ) -> None:

        assert (
            len(self.input_files) == 1
        ), "Only the path folder should be given for benchmark plots!"

        fpath = Path(self.input_files[0]).absolute()

        assert fpath.is_dir(), "The input path should be a folder, not a file!"

        benchmark_files = glob((fpath / "benchmark").as_posix() + "*")
        benchmark_files = [Path(file) for file in benchmark_files if dim in file]

        result_omp = {}
        result_mpi = {}
        result_hybrid = {}
        result_serial = {}

        Ntau = Nleft = Nright = None

        for file in benchmark_files:
            kname = file.name.rsplit(".", maxsplit=1)[0].split("_")[2]
            with open(file, "r") as f:
                tmp_dict = json.load(f)
            mean_time = 0
            repetitions = tmp_dict["Repetitions"]
            Ntau = tmp_dict["Ntau"]
            Nleft = tmp_dict["NLeft"]
            Nright = tmp_dict["NRight"]

            newton_keys = [key for key in tmp_dict["0"].keys() if "NewtonStep" in key]
            assemble_keys = [
                key for key in tmp_dict["0"].keys() if "AssembleStep" in key
            ]
            mean_time = np.mean(
                [tmp_dict[f"{i}"]["OverallTime"] for i in range(repetitions)]
            )
            mean_newton_time = {key: 0 for key in newton_keys}
            mean_assemble_time = {key: 0 for key in assemble_keys}
            for j, (nkey, akey) in product(
                range(repetitions), zip(newton_keys, assemble_keys)
            ):
                mean_newton_time[nkey] += tmp_dict[f"{j}"][nkey] / repetitions
                mean_assemble_time[akey] += tmp_dict[f"{j}"][akey] / repetitions

            overall_newton_time = np.sum([n for n in mean_newton_time.values()])
            overall_assemble_time = np.sum([a for a in mean_assemble_time.values()])

            match kname:
                case "OpenMP":
                    threads = tmp_dict["Threads"]
                    result_omp[threads] = mean_time
                case "MPI":
                    cores = tmp_dict["Cores"]
                    result_mpi[cores] = mean_time
                case "Hybrid":
                    cores = tmp_dict["Cores"]
                    threads = tmp_dict["Threads"]
                    result_hybrid[int(cores * threads)] = mean_time
                case "Serial":
                    result_serial["par_frac"] = overall_assemble_time / mean_time
                    result_serial["t"] = mean_time
                case "_":
                    raise ValueError(
                        f"{kname} is not defined in the evaluation algorithm!"
                    )

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Cores [-]", fontsize=28, labelpad=10)
        if rel_su:
            ax.set_ylabel("Rel. Efficiency [-]", fontsize=28, labelpad=10)
        else:
            ax.set_ylabel("Efficiency [-]", fontsize=28, labelpad=10)
        max_speed = 0
        max_procs = 0
        num = 0
        amdahl_law = lambda c, f: 1 / ((1 - f) + f / c)

        markers = ["o", "s", "^", "D", "v", ">", "<", "P", "*"]
        colors = ["#0F52BA", "#9B111E", "#0E7C61"]

        for i, (dic, kind) in enumerate(
            zip([result_omp, result_mpi, result_hybrid], ["OpenMP", "MPI", "Hybrid"])
        ):

            if dic:
                num += 1
                procs = np.array([key for key in dic.keys()])
                args_procs = np.argsort(procs)
                procs = procs[args_procs]
                max_procs = procs.max() if procs.max() > max_procs else max_procs
                overall_times = np.array([val for val in dic.values()])[args_procs]

                if rel_su:
                    overall_times = (1 / overall_times[0] * overall_times) ** (
                        -1
                    ) / amdahl_law(procs, result_serial["par_frac"])
                else:
                    overall_times = (1 / result_serial["t"] * overall_times) ** (
                        -1
                    ) / procs
                    overall_times[0] = 1.0
                max_speed = (
                    overall_times.max()
                    if overall_times.max() > max_speed
                    else max_speed
                )

                ax.plot(
                    procs,
                    overall_times,
                    label=kind,
                    ls="-",
                    marker=markers[i],
                    color=colors[i],
                )

                print(
                    f"{kind}, {Ntau=}, Nleft={Nright=}, #cores: {procs}, Efficiency: {overall_times}"
                )

        cores_max_exp = int(np.ceil(np.log2(max_procs)))
        cores = np.logspace(0, cores_max_exp, base=2, num=100)

        ax.set_xlim(1, 2**cores_max_exp)
        ax.set_ylim(0, 1)
        ax.set_xticks([2**i for i in range(cores_max_exp + 1) if i % 2 == 0])
        ax.set_xticklabels(
            [f"$2^{{{i}}}$" for i in range(cores_max_exp + 1) if i % 2 == 0]
        )
        ax.set_yticks(np.arange(0, 1.05, 0.25))
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

        ax.grid(color="grey", which="major", linestyle="-", linewidth=0.5, alpha=0.2)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncols=num + 1,
            frameon=False,
            fontsize=20,
        )

        plt.tight_layout()

        if save_fpath:
            save_name = Path(
                save_fpath.name.split(".")[0]
                + f"_efficiency_{Ntau}_{Nleft}"
                + "."
                + save_fpath.name.split(".")[-1]
            )
            fig.savefig((save_fpath.parent / save_name).as_posix())

        plt.show()


# ============================ 5. Main Calculations ==============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CLI for visually representations of results from critical collapse simulation."
    )

    parser.add_argument(
        "-i",
        "--input_files",
        type=str,
        nargs="+",
        default=[
            "../data/simulation_convergence_base_3R.json",
            "../data/simulation_convergence_xcut02_3R.json",
            "../data/simulation_convergence_xcut04_3R.json",
            "../data/simulation_convergence_xcut12_3R.json",
            "../data/simulation_convergence_xcut14_3R.json",
        ],
        help="Path to the input JSON result files.",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default="data/cc_plot.pdf",
        help="Path to save the generated plots with general prefix for the files and correct file ending.",
    )
    parser.add_argument(
        "-e",
        "--experimental_data",
        type=str,
        default="",
        help="Path to experimental data to compare.",
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        default="convergence",
        choices=[
            "convergence",
            "fields",
            "fields_paper",
            "initial_data",
            "echoing_period",
            "benchmark",
            "mismatch_layer_finder",
            "theoretical_speedup",
            "efficiency",
        ],
        help="Type of plot which should be produced.",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="",
        choices=["3R", "differences", "vminu", "rel_su"],
        help="Specifying the kind of plot.",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=str,
        default="",
        help="Dimension which should be postprocessed.",
    )
    parser.add_argument(
        "-s",
        "--single_plots",
        action="store_true",
        help="Set for single instead of grid plots.",
    )

    args = parser.parse_args()

    if args.spec:
        match (args.kind, args.spec):
            case ("convergence", "3R"):
                pass
            case ("initial_data", "differences") | ("echoing_period", "differences"):
                pass
            case ("mismatch_layer_finder", "vminu"):
                pass
            case ("benchmark", "rel_su") | ("efficiency", "rel_su"):
                pass
            case _:
                raise ValueError(
                    f"{args.kind} and {args.spec} is not a valid combination."
                )

    plotter = ResultPlotter(args.input_files)

    plotter.plot(
        args.kind,
        args.output_name,
        single_plots=args.single_plots,
        experimental_data=args.experimental_data,
        dim=args.dim,
        spec=args.spec,
    )
