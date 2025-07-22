#!/bin/usr/python3

# ============================ 1. Importing Libraries ============================

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource
from itertools import product
from decimal import Decimal
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy
import argparse

# ============================ 2. Defining Parameters ============================

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.minor.size'] = 3

# Activate LaTeX style if supported
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ============================ 3. Defining Functions =============================

def round_up(val: float):
    """
    Round up a float to the next "nice" power-of-10-aligned multiple.
    """
    if val == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(val))))
    base = 10 ** exponent
    return np.ceil(val / base) * base

def round_down(val: float):
    """
    Round down a float to the previous "nice" power-of-10-aligned multiple.
    """
    if val == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(val))))
    base = 10 ** exponent
    return np.floor(val / base) * base

def round_limits(zmin: float, zmax: float):
    """
    Round zmin down and zmax up to nearest power-of-10-aligned "nice" bounds.
    """
    return round_down(zmin), round_up(zmax)

# ============================ 4. Defining Classes ===============================

class ResultPlotter:

    def __init__(self, input_files: list=[]):
        self.input_files = input_files

        for i, file in enumerate(self.input_files):
            self.input_files[i] = Path(file).absolute()

    def plot(self, kind: str, save_fpath: str=None, single_plots: bool=False, dim: str=None) -> None:
        
        if save_fpath:
            save_fpath = Path(save_fpath).absolute()
            if not save_fpath.exists(): save_fpath.parent.mkdir(parents=True, exist_ok=True)

        match kind:
            case "convergence":
                self.plot_convergence(save_fpath, single_plots)
            case "fields":
                self.plot_fields(save_fpath, single_plots)
            case "initial_data":
                self.plot_initial_data(save_fpath, dim)
            case "echoing_period":
                pass
            case "benchmark":
                pass
            case _:
                raise ValueError(f'{kind} is not a valid keyword for creating a plot.')
            
    def plot_convergence(self, save_fpath: Path, single_plots: bool) -> None:

        assert len(self.input_files) == 5, "There should be 5 files supplied to create" \
                                         + " the convergence plot."
        
        result_dict = {}

        for file in self.input_files:
            name = file.name.split('.')[0].split('_')[-1]
            with open(file.as_posix(), 'r') as f:
                result_dict[name] = json.load(f)
            assert result_dict[name]['Converged'], f'{name} simulation is not converged, plots cannot be produced.'
            result_dict[name]["tau"] = np.linspace(0, result_dict[name]['Initial_Conditions']['Delta'], num=result_dict[name]["Ntau"])
        
        dim = result_dict['base']['Dim']

        for side in ['left', 'right']:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(32,20))
            
            if side=='left':
                fig.suptitle(f'Left Convergence for D={dim:.3f}', fontsize=25)
                tau = np.array(result_dict['base']['tau'])
                fc1 = np.array(result_dict['xcut02']['Initial_Conditions']['fc']) - np.array(result_dict['base']['Initial_Conditions']['fc'])
                fc2 = 2**(-6)*(np.array(result_dict['xcut04']['Initial_Conditions']['fc']) - np.array(result_dict['xcut02']['Initial_Conditions']['fc']))
                psic1 = np.array(result_dict['xcut02']['Initial_Conditions']['psic']) - np.array(result_dict['base']['Initial_Conditions']['psic'])
                psic2 = 2**(-6)*(np.array(result_dict['xcut04']['Initial_Conditions']['psic']) - np.array(result_dict['xcut02']['Initial_Conditions']['psic']))
                Up1 = np.array(result_dict['xcut02']['Initial_Conditions']['Up']) - np.array(result_dict['base']['Initial_Conditions']['Up'])
                Up2 = 2**(-6)*(np.array(result_dict['xcut04']['Initial_Conditions']['Up']) - np.array(result_dict['xcut02']['Initial_Conditions']['Up']))

                modes = np.fft.rfftfreq(tau.size, d=tau[1]-tau[0])[:int(tau.size//4)]
                fc1_k = np.abs(np.fft.rfft(result_dict['xcut02']['Initial_Conditions']['fc']) \
                             - np.fft.rfft(result_dict['base']['Initial_Conditions']['fc']))[:int(tau.size//4)]
                fc2_k = 2**(-6)*np.abs(np.fft.rfft(result_dict['xcut04']['Initial_Conditions']['fc']) \
                             - np.fft.rfft(result_dict['xcut02']['Initial_Conditions']['fc']))[:int(tau.size//4)]
                psic1_k = np.abs(np.fft.rfft(result_dict['xcut02']['Initial_Conditions']['psic']) \
                             - np.fft.rfft(result_dict['base']['Initial_Conditions']['psic']))[:int(tau.size//4)]
                psic2_k = 2**(-6)*np.abs(np.fft.rfft(result_dict['xcut04']['Initial_Conditions']['psic']) \
                             - np.fft.rfft(result_dict['xcut02']['Initial_Conditions']['psic']))[:int(tau.size//4)]
                Up1_k = np.abs(np.fft.rfft(result_dict['xcut02']['Initial_Conditions']['Up']) \
                             - np.fft.rfft(result_dict['base']['Initial_Conditions']['Up']))[:int(tau.size//4)]
                Up2_k = 2**(-6)*np.abs(np.fft.rfft(result_dict['xcut04']['Initial_Conditions']['Up']) \
                             - np.fft.rfft(result_dict['xcut02']['Initial_Conditions']['Up']))[:int(tau.size//4)]
               

                axes[0,0].plot(tau, fc1, color='#006699', label=r'$f_c(2x_L) - f_c(x_L)$')
                axes[0,0].plot(tau, fc2, color='#511D66', label=r'$2^{-6} \left(f_c(4x_L) - f_c(2x_L) \right)$')
                axes[0,1].plot(tau, psic1, color='#006699', label=r'$\psi_c(2x_L) - \psi_c(x_L)$')
                axes[0,1].plot(tau, psic2, color='#511D66', label=r'$2^{-6} \left(\psi_c(4x_L) - \psi_c(2x_L) \right)$')
                axes[0,2].plot(tau, Up1, color='#006699', label=r'$U_p(2x_L) - U_p(x_L)$')
                axes[0,2].plot(tau, Up2, color='#511D66', label=r'$2^{-6} \left(U_p(4x_L) - U_p(2x_L) \right)$')

                axes[1,0].semilogy(list(range(modes.size)), fc1_k, color='#006699',
                                   label=r'$|\hat{f}^k_c(2x_L) - \hat{f}^k_c(x_L)|$', marker='o', markersize=3, ls='None')
                axes[1,0].semilogy(list(range(modes.size)), fc2_k, color='#511D66',
                                   label=r'$2^{-6} \left|\hat{f}^k_c(4x_L) - \hat{f}^k_c(2x_L) \right|$', marker='o', markersize=3, ls='None')
                axes[1,1].semilogy(list(range(modes.size)), psic1_k, color='#006699',
                                   label=r'$|\hat{\psi}^k_c(2x_L) - \hat{\psi}^k_c(x_L)|$', marker='o', markersize=3, ls='None')
                axes[1,1].semilogy(list(range(modes.size)), psic2_k, color='#511D66',
                                   label=r'$2^{-6} \left|\hat{\psi}^k_c(4x_L) - \hat{\psi}^k_c(2x_L) \right|$', marker='o', markersize=3, ls='None')
                axes[1,2].semilogy(list(range(modes.size)), Up1_k, color='#006699',
                                   label=r'$|\hat{U}^k_p(2x_L) - \hat{U}^k_p(x_L)|$', marker='o', markersize=3, ls='None')
                axes[1,2].semilogy(list(range(modes.size)), Up2_k, color='#511D66',
                                   label=r'$2^{-6} \left|\hat{U}^k_p(4x_L) - \hat{U}^k_p(2x_L) \right|$', marker='o', markersize=3, ls='None')

                for axs,i in product(axes, range(3)):

                    axs[i].grid(True, which='major', axis='both', color='gray',
                                ls=':', lw=0.5)
                    axs[i].legend(loc='upper center', fontsize=18, ncol=2,
                                    bbox_to_anchor=(0.5, 1.12), frameon=False)

                    axs[i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))

                    if axs[i] in axes[0]:
                        axs[i].set_xlim(0, tau[-1])
                        axs[i].set_xlabel(r'$\tau$', fontsize=30)
                        axs[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
                    
                    elif axs[i] in axes[1]:
                        axs[i].set_xlim(0, modes.size)
                        axs[i].set_xlabel(r'$k$', fontsize=30)


            elif side=='right':
                fig.suptitle(f'Right Convergence for D={dim:.3f}', fontsize=25)
                tau = np.array(result_dict['base']['tau'])
                fc1 = np.array(result_dict['xcut12']['Initial_Conditions']['fc']) - np.array(result_dict['base']['Initial_Conditions']['fc'])
                fc2 = 2**(-3)*(np.array(result_dict['xcut14']['Initial_Conditions']['fc']) - np.array(result_dict['xcut12']['Initial_Conditions']['fc']))
                psic1 = np.array(result_dict['xcut12']['Initial_Conditions']['psic']) - np.array(result_dict['base']['Initial_Conditions']['psic'])
                psic2 = 2**(-3)*(np.array(result_dict['xcut14']['Initial_Conditions']['psic']) - np.array(result_dict['xcut12']['Initial_Conditions']['psic']))
                Up1 = result_dict['xcut12']['Initial_Conditions']['Up'] - np.array(result_dict['base']['Initial_Conditions']['Up'])
                Up2 = 2**(-3)*(np.array(result_dict['xcut14']['Initial_Conditions']['Up']) - np.array(result_dict['xcut12']['Initial_Conditions']['Up']))

                modes = np.fft.rfftfreq(tau.size, d=tau[1]-tau[0])[:int(tau.size//4)]
                fc1_k = np.abs(np.fft.rfft(result_dict['xcut12']['Initial_Conditions']['fc']) \
                             - np.fft.rfft(result_dict['base']['Initial_Conditions']['fc']))[:int(tau.size//4)]
                fc2_k = 2**(-3)*np.abs(np.fft.rfft(result_dict['xcut14']['Initial_Conditions']['fc']) \
                             - np.fft.rfft(result_dict['xcut12']['Initial_Conditions']['fc']))[:int(tau.size//4)]
                psic1_k = np.abs(np.fft.rfft(result_dict['xcut12']['Initial_Conditions']['psic']) \
                             - np.fft.rfft(result_dict['base']['Initial_Conditions']['psic']))[:int(tau.size//4)]
                psic2_k = 2**(-3)*np.abs(np.fft.rfft(result_dict['xcut14']['Initial_Conditions']['psic']) \
                             - np.fft.rfft(result_dict['xcut12']['Initial_Conditions']['psic']))[:int(tau.size//4)]
                Up1_k = np.abs(np.fft.rfft(result_dict['xcut12']['Initial_Conditions']['Up']) \
                             - np.fft.rfft(result_dict['base']['Initial_Conditions']['Up']))[:int(tau.size//4)]
                Up2_k = 2**(-3)*np.abs(np.fft.rfft(result_dict['xcut14']['Initial_Conditions']['Up']) \
                             - np.fft.rfft(result_dict['xcut12']['Initial_Conditions']['Up']))[:int(tau.size//4)]
               

                axes[0,0].plot(tau, fc1, color='#006699', label=r'$f_c(2x_R) - f_c(x_R)$')
                axes[0,0].plot(tau, fc2, color='#511D66', label=r'$2^{-3} \left(f_c(4x_R) - f_c(2x_R) \right)$')
                axes[0,1].plot(tau, psic1, color='#006699', label=r'$\psi_c(2x_L) - \psi_c(x_R)$')
                axes[0,1].plot(tau, psic2, color='#511D66', label=r'$2^{-3} \left(\psi_c(4x_R) - \psi_c(2x_R) \right)$')
                axes[0,2].plot(tau, Up1, color='#006699', label=r'$U_p(2x_R) - U_p(x_R)$')
                axes[0,2].plot(tau, Up2, color='#511D66', label=r'$2^{-3} \left(U_p(4x_R) - U_p(2x_R) \right)$')

                axes[1,0].semilogy(list(range(modes.size)), fc1_k, color='#006699',
                                   label=r'$|\hat{f}^k_c(2x_R) - \hat{f}^k_c(x_R)|$', marker='o', markersize=3, ls='None')
                axes[1,0].semilogy(list(range(modes.size)), fc2_k, color="#242324",
                                   label=r'$2^{-3} \left|\hat{f}^k_c(4x_R) - \hat{f}^k_c(2x_R) \right|$', marker='o', markersize=3, ls='None')
                axes[1,1].semilogy(list(range(modes.size)), psic1_k, color='#006699',
                                   label=r'$|\hat{\psi}^k_c(2x_R) - \hat{\psi}^k_c(x_R)|$', marker='o', markersize=3, ls='None')
                axes[1,1].semilogy(list(range(modes.size)), psic2_k, color='#511D66',
                                   label=r'$2^{-3} \left|\hat{\psi}^k_c(4x_R) - \hat{\psi}^k_c(2x_R) \right|$', marker='o', markersize=3, ls='None')
                axes[1,2].semilogy(list(range(modes.size)), Up1_k, color='#006699',
                                   label=r'$|\hat{U}^k_p(2x_R) - \hat{U}^k_p(x_R)|$', marker='o', markersize=3, ls='None')
                axes[1,2].semilogy(list(range(modes.size)), Up2_k, color='#511D66',
                                   label=r'$2^{-3} \left|\hat{U}^k_p(4x_R) - \hat{U}^k_p(2x_R) \right|$', marker='o', markersize=3, ls='None')

                for axs,i in product(axes, range(3)):

                    axs[i].grid(True, which='major', axis='both', color='gray',
                                ls=':', lw=0.5)
                    axs[i].legend(loc='upper center', fontsize=18, ncol=2,
                                    bbox_to_anchor=(0.5, 1.12), frameon=False)

                    axs[i].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))

                    if axs[i] in axes[0]:
                        axs[i].set_xlim(0, tau[-1])
                        axs[i].set_xlabel(r'$\tau$', fontsize=30)
                        axs[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
                    
                    elif axs[i] in axes[1]:
                        axs[i].set_xlim(0, modes.size)
                        axs[i].set_xlabel(r'$k$', fontsize=30)

            if save_fpath:
                save_name = Path(save_fpath.name.split('.')[0] + f'_convergence_{side}' + '.' + save_fpath.name.split('.')[-1])
                plt.savefig((save_fpath.parent / save_name).as_posix())
            
            plt.tight_layout()
            plt.show()
    
    def plot_fields(self, save_fpath: Path, single_plots: bool) -> None:
        
        result_dict = {}

        for file in self.input_files:
            _, dim, step = file.name.rsplit('.', maxsplit=1)[0].split('_')
            with open(file.as_posix(), 'r') as f:
                result_dict[step] = json.load(f)

            result_dict[step]["x"] = np.array(sorted(list(result_dict[step].keys()))[:-1])
            result_dict[step]["tau"] = np.linspace(
                0, result_dict[step]['Delta'],
                num=len(result_dict[step][str(result_dict[step]["x"][0])]['A'])
            )

            As, Us, Vs, fs = [], [], [], []
            for x in result_dict[step]["x"]:
                x_str = str(x)
                As.append(result_dict[step][x_str]['A'])
                Us.append(result_dict[step][x_str]['U'])
                Vs.append(result_dict[step][x_str]['V'])
                fs.append(result_dict[step][x_str]['f'])
                del result_dict[step][x_str]

            result_dict[step]['a'] = np.array(As).T
            result_dict[step]['f'] = np.array(fs).T
            result_dict[step]['U'] = np.array(Us).T
            result_dict[step]['V'] = np.array(Vs).T

            result_dict[step]["x"], result_dict[step]["tau"] = np.meshgrid(
                np.array(result_dict[step]["x"], dtype=np.float64),
                result_dict[step]["tau"]
            )

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 25), subplot_kw={'projection': '3d'})

            # fig = plt.figure(figsize=(25, 35))
            # gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.4)

            # ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            # ax2 = fig.add_subplot(gs[0, 1], projection='3d')
            # ax3 = fig.add_subplot(gs[1, 0], projection='3d')
            # ax4 = fig.add_subplot(gs[1, 1], projection='3d')

            # axes = [ax1, ax2, ax3, ax4]

            fields = ['a', 'f', 'U', 'V']
            cmaps = [mpl.cm.Blues, mpl.cm.Greens, mpl.cm.Reds, mpl.cm.Purples]
            axtitles = [r'$a(r,\tau)$', r'$f(r,\tau)$',r'$U(r,\tau)$', r'$V(r,\tau)$']

            for ax, field, cmap, axtitle in zip(axes.flatten(), fields, cmaps, axtitles):
                
                x = result_dict[step]["x"]
                t = result_dict[step]["tau"]
                data = result_dict[step][field]
                
                xmin, xmax = round_limits(x.min(), x.max())
                ymin, ymax = round_limits(t.min(), t.max())
                zmin, zmax = round_limits(data.min(), data.max())
                
                ls = LightSource(azdeg=315, altdeg=45)
                rgb = ls.shade(data, cmap=cmap, vert_exag=0.1, blend_mode='soft')

                ax.plot_surface(
                    x,
                    t,
                    data,
                    facecolors=rgb,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True
                )

                ax.zaxis.set_rotate_label(False)
                ax.set_xlabel(r'$r$', fontsize=28, labelpad=20)
                ax.set_ylabel(r'$\tau$', fontsize=28, labelpad=20)
                ax.set_zlabel(f'${axtitle}$', fontsize=28, labelpad=20, rotation=90)

                ax.tick_params(axis='x', pad=10, labelsize=22)
                ax.tick_params(axis='y', pad=10, labelsize=22)
                ax.tick_params(axis='z', pad=10, labelsize=22, width=1.5, length=6)

                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))  
                ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))  
                ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

                xticks = ax.get_xticks()
                xtick_labels = [f"{tick:.1f}" if tick == 0 else str(tick) for tick in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels)

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_zlim(zmin, zmax)

                ax.view_init(elev=25, azim=-140)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

            plt.subplots_adjust(left=0.1, right=0.90, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
            #plt.tight_layout()

            if save_fpath:
                save_name = Path(save_fpath.name.split('.')[0] + f'_fields_{dim}_{step}' + '.' + save_fpath.name.split('.')[-1])
                plt.savefig((save_fpath.parent / save_name).as_posix())

            plt.show()


    def plot_initial_data(self, save_fpath: Path, dim: str) -> None:

        fpath = Path(self.input_files[0])
        result_dict = {}
        with open(fpath.as_posix(), 'r') as f:
            result_dict = json.load(f)
        
        if dim: 
            result_dict = result_dict[dim]

        assert result_dict['Converged'], "Data has to be converged to be displayed!"

        t = np.linspace(0, result_dict['Initial_Conditions']['Delta'], num=result_dict['Ntau'])
        fc = np.array(result_dict['Initial_Conditions']['fc'], dtype=np.float64)
        psic = np.array(result_dict['Initial_Conditions']['psic'], dtype=np.float64)
        Up = np.array(result_dict['Initial_Conditions']['Up'], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(10,8))

        ax.plot(t, fc, label=r'$f_c$', ls='-', color='#006699')
        ax.plot(t, Up, label=r'$U_p$', ls='-.', color='#006699')

        ax2 = ax.twinx()
        ax2.plot(t, psic, label=r'$\psi_c$', ls='--', color='#511D66')

        for label in ax.get_yticklabels():
            label.set_color('#006699')
        
        for label in ax2.get_yticklabels():
            label.set_color('#511D66')
        
        fcmin, fcmax = round_limits(fc.min(), fc.max())
        Upmin, Upmax = round_limits(Up.min(), Up.max())
        fcmin = fcmin if fcmin < Upmin else Upmin
        fcmax = fcmax if fcmax > Upmax else Upmax
        psicmin, psicmax = round_limits(psic.min(), psic.max())

        ax.set_xlim(0, np.ceil(t.max()*10)/10)
        ax.set_ylim(fcmin, fcmax)
        ax2.set_ylim(psicmin, psicmax)

        ax.set_xlabel(r'$\tau$', fontsize=28)
        
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))  
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))

        ax.grid(color='grey', which='major', linestyle='-', linewidth=0.5, alpha=0.4)
        #ax.grid(color='grey', which='minor', linestyle=':', linewidth=0.25)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax.legend(
            lines + lines2,
            labels + labels2,
            loc='upper center',
            fontsize=20,
            ncol=3,
            bbox_to_anchor=(0.5, 1.10),
            frameon=False
        )

        plt.tight_layout()

        if save_fpath:
            save_name = Path(save_fpath.name.split('.')[0] 
                             + f'_initial-data_{dim}' + '.' 
                             + save_fpath.name.split('.')[-1])
            plt.savefig((save_fpath.parent / save_name).as_posix())

        plt.show()


    def plot_echoing_period(self, save_fpath: Path) -> None:
        pass

    def plot_benchmark(self, save_fpath: Path) -> None:
        pass


# ============================ 5. Main Calculations ==============================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='CLI for visually representations of results from critical collapse simulation.'
    )

    parser.add_argument(
        '-i', '--input_files',
        type=str,
        nargs='+',
        default=["data/simulation_convergence_base.json",
                 "data/simulation_convergence_xcut02.json",
                 "data/simulation_convergence_xcut04.json",
                 "data/simulation_convergence_xcut12.json",
                 "data/simulation_convergence_xcut14.json"],
        help='Path to the input JSON result files.'
    )
    parser.add_argument(
        '-o', '--output_name',
        type=str,
        default="data/cc_plot.pdf",
        help='Path to save the generated plots.'
    )
    parser.add_argument(
        '-k', '--kind',
        type=str,
        default='convergence',
        choices=['convergence', 'fields', 'initial_data'],
        help='Type of plot which should be produced.'
        )
    parser.add_argument(
        '-d', '--dim',
        type=float,
        default='',
        help='Dimension which should be postprocessed'
    )
    
    args = parser.parse_args()

    plotter = ResultPlotter(args.input_files)

    plotter.plot(args.kind, args.output_name, dim=f'{args.dim:.3f}')
