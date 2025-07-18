#!/bin/usr/python3

# ============================ 1. Importing Libraries ============================

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# ============================ 3. Defining Classes ===============================

class ResultPlotter:

    def __init__(self, input_files: list=[]):
        self.input_files = input_files

        for i, file in enumerate(self.input_files):
            self.input_files[i] = Path(file).absolute()

    def plot(self, kind: str, save_fpath: str=None) -> None:
        
        if save_fpath:
            save_fpath = Path(save_fpath).absolute()
            if not save_fpath.exists(): save_fpath.parent.mkdir(parents=True, exist_ok=True)

        match kind:
            case "convergence":
                self.plot_convergence(save_fpath)
            case _:
                raise ValueError(f'{kind} is not a valid keyword for creating a plot.')
            
    def plot_convergence(self, save_fpath: Path) -> None:

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
                axes[1,0].semilogy(list(range(modes.size)), fc2_k, color='#511D66',
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



# ============================ 4. Defining Functions =============================

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
        choices=['convergence'],
        help='Type of plot which should be produced.'
        )
    
    args = parser.parse_args()

    print(args.input_files)

    plotter = ResultPlotter(args.input_files)

    plotter.plot(args.kind, args.output_name)
