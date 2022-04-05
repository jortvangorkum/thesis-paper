import pandas as pd

def parse_time_data_file(time_memory_path: str, file_name: str) -> pd.DataFrame:
    df_data_time = pd.read_csv(f"{time_memory_path}/{file_name}.csv") 

    df_parsed_data_time = pd.DataFrame()

    df_parsed_data_time[["Benchmark", "Amount Nodes"]] = df_data_time["Name"].str.split('/', expand=True)
    df_parsed_data_time["Amount Nodes"] = pd.to_numeric(df_parsed_data_time["Amount Nodes"])
    df_parsed_data_time["Execution Time"] = df_data_time["MeanUB"]

    return df_parsed_data_time
