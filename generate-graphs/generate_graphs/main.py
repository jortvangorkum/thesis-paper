import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats

regex = r"benchmarking (.*?)\/(\d*)(?:(?:.|\n)*?)iters(?:(?:\s)*)(\d+\.?\d+e?\d*)"

DATA_MEMORY_PATH = './data/memory'

def parse_memory_data_file(file_name: str) -> pd.DataFrame:
    with open(f"{DATA_MEMORY_PATH}/{file_name}.txt", 'r') as data_memory_file:
        data_memory_contents = data_memory_file.read()
        matches = re.finditer(regex, data_memory_contents)
        df_data_memory = pd.DataFrame(columns=['Benchmark', 'Amount Nodes', 'Allocated Bytes'])

        for match in matches:
            benchmark_name  = match.group(1)
            amount_nodes    = match.group(2)
            allocated_bytes = match.group(3)

            if (benchmark_name is None or amount_nodes is None or allocated_bytes is None):
                raise Exception("Cannot correctly convert data memory benchmark to values")

            df_data_memory.loc[len(df_data_memory.index)] = [benchmark_name, int(amount_nodes), float(allocated_bytes)]

        return df_data_memory

def plot_benchmark(data: pd.DataFrame) -> Axes:
    (slope, intercept, r_value, _, _) = stats.linregress(data["Amount Nodes"], data["Allocated Bytes"])
    
    ax = sns.regplot(
        x="Amount Nodes", 
        y="Allocated Bytes", 
        data=data, 
        line_kws={
            'label': "y={0:.1f}x+{1:.1f}, R²={2:.2f}".format(slope, intercept, r_value * r_value)
            }
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()

    return ax


if __name__ == "__main__":
    df_data_memory = parse_memory_data_file('run-0')
    print(df_data_memory)

    sns.set_palette("pastel")

    ax_data_memory = plot_benchmark(df_data_memory)
    plt.show()
