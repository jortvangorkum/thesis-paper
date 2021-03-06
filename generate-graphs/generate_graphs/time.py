import pandas as pd

def parse_time_data_file(time_memory_path: str, file_name: str) -> pd.DataFrame:
    df_data_time = pd.read_csv(f"{time_memory_path}/{file_name}.csv") 

    df_parsed_data_time = pd.DataFrame()

    df_parsed_data_time[["Name", "Case", "Iterations", "Benchmark", "Amount Nodes"]] = df_data_time["Name"].str.split('/', expand=True)
    df_parsed_data_time["Amount Nodes"] = pd.to_numeric(df_parsed_data_time["Amount Nodes"])
    df_parsed_data_time["Execution Time"] = df_data_time["MeanUB"]
    df_parsed_data_time["Standard Deviation"] = df_data_time["Stddev"]

    print(df_parsed_data_time)

    return df_parsed_data_time
