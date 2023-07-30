from tqdm.auto import tqdm
from multiprocessing import Pool
import pandas as pd
import subprocess

def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return x

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1)
    """
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    processed = []
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
    return processed

def format_code(cpp_code):
    """
        remember to run `apt install clang-format`, version 14.0.0-1ubuntu1
    """
    preprocessed_code = subprocess.check_output(['clang-format', '-style=Google'], input=cpp_code, universal_newlines=True)
    return preprocessed_code
