import os
import re
from os.path import join

import numpy as np
import pandas as pd
import pathlib
import sqlite3


# Data loaders
def load_csv_files(
    csv_paths: list[str],
    focus_columns=None,
    drop_duplicates=True,
    dropna=True,
    shuffle=False,
    single_output=False,
):
    """
    Combine csv data into a single dataframe and checks for duplicate records.

    """

    indices = []

    for i, path in enumerate(csv_paths):
        print(f"Loading '{os.path.basename(path)}'")

        df = pd.read_csv(path)

        columns = df.columns

        print(f"Number of records in {path}: {df.shape[0]}")

        if i == 0:
            df_full = df
            columns_base = columns

            indices.append([0, len(df)])

        else:
            if not np.array_equal(columns, columns_base):
                raise (Exception("Columns do not match"))

            indices.append([len(df_full), len(df_full) + len(df)])

            df_full = pd.concat([df_full, df]).reset_index(drop=True)

            print(f"-> Merged!!")

        print("")

    focus_columns = df_full.columns if focus_columns == None else focus_columns

    print(f"Focusing on the following columns: {focus_columns}\n")

    if drop_duplicates:
        print("Searching for duplicate rows in focus columns...")
        total_records = df_full.shape[0]
        df_full.drop_duplicates(subset=focus_columns, keep="first", inplace=True)
        records_dropped = total_records - df_full.shape[0]
        print(
            f"A total of {df_full.shape[0]} records were loaded ({records_dropped} records dropped after duplicate filter)\n"
        )

    if dropna:
        print(f"Seaching for NaN fields in foclus columns...")
        nan_row_count = df_full[df_full[focus_columns].isna().values].shape[0]
        print(f"Rows with NaN values: {nan_row_count}")
        print("Dropping NaN...")
        df_full.dropna(subset=focus_columns, inplace=True)
        print(f"\nFinal total records {df_full.shape[0]}\n")

    if shuffle:
        print("shuffling indices")
        shuffled_indices = np.random.permutation(np.arange(df_full.shape[0]))
        df_full = df_full.iloc[shuffled_indices, :]

    if single_output:
        print("returning a single file...")
        return df_full
    else:
        print(f"returning {len(indices)} files")
        return_list = []

        for i in indices:
            return_list.append(df_full.iloc[i[0] : i[1], :])

        return return_list


# Tokenizer based functions


def token_to_df(tokenized_input: pd.DataFrame, tokenizer):
    df = pd.DataFrame()
    for col in tokenized_input.keys():
        df[col] = tokenized_input[col][0]

    vocab_from_tk = []

    for tk in tokenized_input["input_ids"][0]:
        vocab_from_tk.append(tokenizer.decode(tk, skip_special_tokens=False))

    df["vocab_word"] = vocab_from_tk

    return df


def expand_tokenizer(new_tokens: list[str], tokenizer):
    new_tokens_len = len(new_tokens)

    # check if the tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())

    print(f"Number of tokens processed: {new_tokens_len}")
    print(f"Already existing in vocabulary: {new_tokens_len - len(new_tokens)}")
    print(f"Tokens added to vocavulary: {len(new_tokens)}")

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))

    tokens = tokenizer(" ".join(new_tokens), return_tensors="pt")

    return token_to_df(tokenized_input=tokens, tokenizer=tokenizer)


# Accuracy calcs
def get_string_equality(str1: str, str2: str, delimiter=" "):
    str1 = np.array(str1.split())
    str2 = np.array(str2.split())

    diff = len(str1) - len(str2)

    if diff < 0:
        str1 = np.pad(str1, (0, np.abs(diff)), mode="constant", constant_values="<pad>")
    elif diff > 0:
        str2 = np.pad(str2, (0, diff), mode="constant", constant_values="<pad>")

    tally = np.char.equal(str1, str2)

    return tally.mean()


def get_pd_row_accuracy(row):
    return get_string_equality(row["labels"], row["prediction"])


# Word processing
def replace_first(string: str, old: str, new: str):
    return string.replace(old, new, 1)


def replace_last(string: str, old: str, new: str):
    string = string[::-1]
    old = old[::-1]
    new = new[::-1]

    return string.replace(old, new, 1)[::-1]


def sperarate_substring_by_spaces(sentence, substring, process_char_mid_word=False):
    regex_meta_characters = ["^", "$", "*", "+", "?", ".", "(", ")"]

    regex_char = ""
    for c in list(substring):
        c = ("\\" + c) if c in regex_meta_characters else c
        regex_char = regex_char + c

    sentence_list = []

    for w in sentence.split():
        if substring not in w:
            sentence_list.append(w)
        elif w == substring:
            sentence_list.append(w)
        else:
            # Starting pos replace
            if w.startswith(substring):
                w = replace_first(w, substring, substring + " ")

            # End pos replace
            if w.endswith(substring):
                w = replace_last(w, substring, " " + substring)

            # Mid pos replace
            if process_char_mid_word:
                pattern = r"(\w)" + regex_char + r"(\w)"
                repl = r" " + substring + r" "
                repl_pattern = r"\1" + repl + r"\2"
                w = re.sub(pattern=pattern, repl=repl_pattern, string=w)

            sentence_list.append(w)

    return " ".join(sentence_list)


def get_vega_zero_table(row):
    vega_zero = row["vega_zero"].split()

    if vega_zero[2] != "data":
        raise Exception("Expected 'data' to be at index 2 of the vega_zero query")

    return vega_zero[3]


def get_vega_zero_used_columns(row):
    if pd.isna(row["mentioned_columns"]):
        return ""
    else:
        columns_used = []
        mentioned_columns = str(row["mentioned_columns"]).split()

        for c in mentioned_columns:
            if c == "NaN":
                print(c)

            if c in row["vega_zero"]:
                columns_used.append(c)

        return " ".join(columns_used)


def get_columns_used_in_vz_query(row, ref_column="mentioned_columns"):
    columns_used = []
    for c in row[ref_column]:
        if c in row["vega_zero"]:
            columns_used.append(c)

    return columns_used


def get_ncNet_table_used_col_info(df):
    df["table_used"] = df.apply(get_vega_zero_table, axis=1)

    df["mentioned_columns"] = df["mentioned_columns"].apply(
        lambda var: "" if pd.isna(var) else var
    )

    df["mentioned_columns_count"] = df["mentioned_columns"].apply(
        lambda var: len(str(var).split())
    )

    df["columns_used"] = df.apply(get_vega_zero_used_columns, axis=1)

    df["columns_used_count"] = df["columns_used"].apply(
        lambda var: len(str(var).split())
    )

    return df


class nvBenchDatabase:
    def __init__(self, path: str, extension: str = ".sqlite") -> None:
        self.directory = pathlib.Path(path)
        self.extension = extension
        self.find_db_files()

    def find_db_files(self):
        path_list = []
        name_list = []
        for file in self.directory.rglob(f"*{self.extension}"):
            name_list.append(file.name.replace(self.extension, ""))
            path_list.append(str(file))

        self.databases_df = pd.DataFrame()
        self.databases_df["db_id"] = name_list
        self.databases_df["db_path"] = path_list

        self.databases_df.sort_values(by="db_id", inplace=True)
        self.databases = self.databases_df["db_id"].values

    def open_connection(self, database):
        path = self.databases_df.loc[self.databases_df["db_id"] == database][
            "db_path"
        ].values[0]

        return sqlite3.connect(path)

    def select_query_pandas(self, database, query):
        error_msg = None
        try:
            connection = self.open_connection(database=database)
            df = pd.read_sql_query(query, connection)
            connection.close()

        except Exception as e:
            df = pd.DataFrame()
            error_msg = e

        finally:
            return df, error_msg

    def get_database_details(self, lower_case=False, include_path=True):
        df = self.databases_df.copy(deep=True)

        success_list = []
        tables_list = []

        for i, db in enumerate(self.databases):
            sql_query = """
                SELECT name FROM sqlite_master  
                WHERE type='table';
            """
            tables, error_msg = self.select_query_pandas(db, sql_query)

            successful_open = True if error_msg == None else False
            success_list.append(successful_open)

            if successful_open:
                if lower_case:
                    tables_list.append(tables["name"].str.lower().to_list())
                else:
                    tables_list.append(tables["name"].to_list())

        df["successful_open"] = success_list
        df["tables"] = tables_list

        if include_path:
            return df
        else:
            return df[df.columns.drop("db_path")]

    def get_database_table_details(self, lower_case=False):
        df_db_details = self.get_database_details(
            lower_case=lower_case, include_path=False
        )

        success_list = []
        db_ids = []
        table_names = []
        column_names = []

        for i, db in enumerate(self.databases):
            db_details = df_db_details.query(f"db_id=='{db}'")
            db_successful_open = db_details["successful_open"].values[0]

            if db_successful_open:
                tables = db_details["tables"].values[0]

                for ii, table in enumerate(tables):
                    sql_query = (
                        f"SELECT name FROM PRAGMA_TABLE_INFO('{table.strip()}');"
                    )
                    cols, error_msg = self.select_query_pandas(db, sql_query)
                    table_successful_open = True if error_msg == None else False

                    db_ids.append(db)
                    table_names.append(table)
                    success_list.append(table_successful_open)

                    if lower_case:
                        column_names.append(cols["name"].str.lower().to_list())
                    else:
                        column_names.append(cols["name"].to_list())

            else:
                db_ids.append(db)
                success_list.apped(db_successful_open)
                table_names.append("")
                column_names.append([])

        df = pd.DataFrame()
        df["db_id"] = db_ids
        df["table_name"] = table_names
        df["column_names"] = column_names
        df["successful_open"] = table_successful_open

        return df


def attach_nvBench_info(df_data: pd.DataFrame, nvBench_db_path: str) -> pd.DataFrame:
    df_data["mentioned_columns"] = df_data["mentioned_columns"].apply(
        lambda var: [] if pd.isna(var) else var.split()
    )

    df_data["table"] = df_data.apply(get_vega_zero_table, axis=1)

    # Load the nvBench database info
    nvBench_Database = nvBenchDatabase(path=nvBench_db_path)
    df_nvBench_db = nvBench_Database.get_database_table_details(lower_case=True)
    df_nvBench_db = df_nvBench_db[df_nvBench_db.columns.drop("successful_open")]
    df_nvBench_db.columns = ["db_id", "table", "nvBench_column_names"]

    # Checking if all the db_ids in ncNet appear on nvBench
    assert set(df_data["db_id"].values).issubset(set(df_nvBench_db["db_id"].values))

    # Merging with ncNet
    df_data = pd.merge(df_data, df_nvBench_db, on=["db_id", "table"], how="left")

    df_data["mentioned_columns_found_in_db"] = df_data.apply(
        lambda row: set(row["mentioned_columns"]).issubset(row["nvBench_column_names"]),
        axis=1,
    )

    # checking if there are any columns where the ncNet table name is not found on nvBench
    assert len(df_data.loc[df_data["nvBench_column_names"].isna()]) == 0

    # Checking if all the columns are found in nvBench db
    assert len(df_data.loc[df_data["mentioned_columns_found_in_db"] == False]) == 0

    # Get columns used in the vega-zero query
    df_data["columns_used"] = df_data.apply(
        get_columns_used_in_vz_query, axis=1, args=(["nvBench_column_names"])
    )

    return df_data[df_data.columns.drop("mentioned_columns_found_in_db")]


def build_vega_zero_source(row, ref_column="mentioned_columns", boost_columns: int = 1):
    question = row["question"]
    template = row["query_template"]
    columns = " ".join(row[ref_column] * boost_columns)
    table = row["table"]
    values = row["mentioned_values"]

    source = f"<N> {question} </N> <C> {template} </C> <D> {table} <COL> {columns} </COL> <VAL> {values} </VAL> </D>"
    return source


if __name__ == "__main__":
    sentence = 'mark point data employees encoding x salary y aggregate none department_id transform filter salary between 8000 and 12000 and commission_pct!= "null" or department_id!= 40'
    print("")
    print("*" * 100)
    print(sentence)
    print(
        sperarate_substring_by_spaces(
            sentence=sentence, substring="!=", process_char_mid_word=True
        )
    )
