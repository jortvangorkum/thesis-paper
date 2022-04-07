from typing import List, Tuple
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
from scipy.optimize import curve_fit
import statsmodels.api as sm
import matplotlib.pyplot as plt


def calculate_rsquared(x: pd.Series, y: pd.Series, f, popt) -> float:
    ss_res   = np.dot((y - f(x, *popt)), (y - f(x, *popt)))
    ymean    = np.mean(y)
    ss_tot   = np.dot((y-ymean), (y-ymean))
    rsquared = 1 - ss_res/ss_tot

    return rsquared

def plot_log_line(x: pd.Series, y: pd.Series) -> None:
    def log_func(x, a, b): 
        return a + b * np.log(x)

    (popt, _) = curve_fit(log_func, x, y)
    rsquared = calculate_rsquared(x, y, log_func, popt)

    ax1 = sns.lineplot(x=x, y=x.apply(lambda i: log_func(i, *popt)))
    
    ax1.legend(title="Equation", loc='upper left', labels=["y={0:.2e} * ln(x) + {1:.2e}, R²={2:.2f}".format(*popt, rsquared)])

def plot_linear_line(x: pd.Series, y: pd.Series) -> None:
    model = sm.OLS(y, x)
    r = model.fit()
    slope = r.params[0]
    rsquared = r.rsquared 

    ax1 = sns.lineplot(x=x, y=slope * x)
    
    ax1.legend(title="Equation", loc='upper left', labels=["y={0:.2e} * x, R²={1:.2f}".format(slope, rsquared)])

def plot_linear_benchmark(df: pd.DataFrame, benchmark_name: str, x_name: str, y_name: str) -> None:
    data = df[df['Benchmark'] == benchmark_name]
    x = data[x_name]
    y = data[y_name]

    plot_linear_line(x, y)
    ax2 = sns.scatterplot(x=x, y=y, label="_nolegend_")

    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.xscale('log')
    plt.yscale('log')

def plot_log_benchmark(df: pd.DataFrame, benchmark_name: str, x_name: str, y_name: str) -> None:
    data = df[df['Benchmark'] == benchmark_name]
    x = data[x_name]
    y = data[y_name]

    plot_log_line(x, y)
    ax2 = sns.scatterplot(x=x, y=y, label="_nolegend_")

    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.xscale('log')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,3), useMathText=True)

def plot_all_benchmarks(df: pd.DataFrame, x_name: str, y_name: str) -> None:
    ax = sns.lineplot(
        data=df,
        x=x_name,
        y=y_name,
        style='Benchmark',
        hue='Benchmark',
        markers=True,
        dashes=False,
    )

    plt.xscale('log')
    plt.yscale('log')


def plot_compare_runs(df_time_run1: pd.DataFrame, df_time_run2: pd.DataFrame, df_mem_run1: pd.DataFrame, df_mem_run2: pd.DataFrame) -> None:
    (fig, ((ax1, ax2), (ax3, ax4))) = plt.subplots(2, 2, sharex=True)  # type: ignore

    ax1.set_title('Run 1')
    ax2.set_title('Run 2')

    l1 = sns.lineplot(
        data=df_mem_run1,
        ax=ax1,
        x='Amount Nodes',
        y='Memory Usage',
        style='Benchmark',
        hue='Benchmark',
        markers=True,
        dashes=False,
    )

    l2 = sns.lineplot(
        data=df_mem_run2,
        ax=ax2,
        x='Amount Nodes',
        y='Memory Usage',
        style='Benchmark',
        hue='Benchmark',
        markers=True,
        dashes=False,
    )

    l3 = sns.lineplot(
        data=df_time_run1,
        ax=ax3,
        x='Amount Nodes',
        y='Execution Time',
        style='Benchmark',
        hue='Benchmark',
        markers=True,
        dashes=False,
    )

    l4 = sns.lineplot(
        data=df_time_run2,
        ax=ax4,
        x='Amount Nodes',
        y='Execution Time',
        style='Benchmark',
        hue='Benchmark',
        markers=True,
        dashes=False,
    )

    for l in [l1, l2, l3, l4]:
        l.set_xscale('log')
        l.set_yscale('log')
    
    for l in [l2, l3, l4]:
        l.get_legend().remove()

    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    fig.tight_layout()