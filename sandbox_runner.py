"""Runs LLM-written pandas code in a child process."""

import ast
import builtins
import io
import json
import sys

import numpy as np
import pandas as pd

BANNED_NAMES = {
    "open", "exec", "eval", "compile", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "__import__",
    "os", "sys", "subprocess", "importlib", "builtins", "shutil", "socket",
}

BANNED_ATTRS = {
    "read_csv", "read_excel", "read_json", "read_pickle", "read_sql",
    "read_parquet", "read_html", "read_table", "read_clipboard",
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet",
    "to_clipboard", "to_hdf", "to_html", "to_feather",
    "load", "save", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "eval", "query", "io", "os", "sys", "system", "popen",
}

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "int", "isinstance", "len", "list", "map", "max", "min",
        "range", "reversed", "round", "set", "sorted", "str", "sum",
        "tuple", "zip", "Exception", "ValueError", "KeyError", "TypeError",
    )
}


def validate_code(code: str):
    """Return an error message, or None if the code passes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Imports are not allowed. df, pd and np are already available."
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return "global/nonlocal are not allowed."
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BANNED_ATTRS:
                return f"Attribute '{node.attr}' is not allowed."
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in BANNED_NAMES:
                return f"Name '{node.id}' is not allowed."
    return None


def emit(obj):
    sys.stdout.write(json.dumps(obj))


def main():
    payload = json.load(sys.stdin)
    code = payload["code"]

    error = validate_code(code)
    if error:
        emit({"ok": False, "error": error})
        return

    df = pd.read_json(io.StringIO(payload["df"]), orient="split", convert_dates=False)
    namespace = {"__builtins__": SAFE_BUILTINS, "df": df, "pd": pd, "np": np}

    try:
        exec(compile(code, "<llm_code>", "exec"), namespace)
        result = namespace.get("result")

        if result is None:
            emit({"ok": False, "error": "Assign the final DataFrame to a variable named `result`."})
            return

        if isinstance(result, pd.Series):
            result = result.reset_index()
        elif not isinstance(result, pd.DataFrame):
            result = pd.DataFrame({"VALUE": [result]})

        # Preserve the DataFrame index.
        # Reset only when the LLM explicitly needs the index as a column.

        # Flatten MultiIndex columns.
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ["_".join(str(p) for p in c if str(p)) for c in result.columns]

        result.columns = [str(c) for c in result.columns]
        emit({"ok": True, "df": result.to_json(orient="split", date_format="iso")})

    except Exception as exc:
        emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()