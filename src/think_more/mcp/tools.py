"""MCP tool implementations using Jedi."""
from pathlib import Path

import jedi


def trace_symbol(
    symbol: str,
    file_path: str,
    project_path: str,
) -> dict:
    """
    Trace a symbol's definitions and references.

    Args:
        symbol: The symbol name to trace
        file_path: Path to the source file
        project_path: Root path of the project

    Returns:
        Dict with definitions, references, and summary
    """
    try:
        source = Path(file_path).read_text()
    except FileNotFoundError:
        return {
            "definitions": [],
            "references": [],
            "summary": f"File not found: {file_path}",
        }

    # Find all occurrences of the symbol in the file
    definitions = []
    references = []

    lines = source.split("\n")
    for line_num, line in enumerate(lines, start=1):
        if symbol not in line:
            continue

        # Find column position
        col = line.find(symbol)
        if col == -1:
            continue

        try:
            script = jedi.Script(source, path=file_path, project=jedi.Project(project_path))

            # Try to get definitions at this position
            names = script.goto(line_num, col)
            for name in names:
                if name.line == line_num:
                    # This is the definition site
                    definitions.append({
                        "file": str(name.module_path) if name.module_path else file_path,
                        "line": name.line,
                        "context": lines[name.line - 1].strip() if name.line <= len(lines) else "",
                    })

            # Get references
            refs = script.get_references(line_num, col)
            for ref in refs:
                if ref.line != line_num or str(ref.module_path) != file_path:
                    references.append({
                        "file": str(ref.module_path) if ref.module_path else file_path,
                        "line": ref.line,
                        "context": "",
                    })
        except Exception:
            pass

    # Deduplicate definitions
    seen_defs = set()
    unique_defs = []
    for d in definitions:
        key = (d["file"], d["line"])
        if key not in seen_defs:
            seen_defs.add(key)
            unique_defs.append(d)

    # Summary
    if unique_defs:
        first_def = unique_defs[0]
        summary = f"{symbol} defined at {first_def['file']}:{first_def['line']}, {len(references)} references"
    else:
        summary = f"{symbol} not found in {file_path}"

    return {
        "definitions": unique_defs,
        "references": references,
        "summary": summary,
    }


def trace_callchain(
    entry_point: str,
    file_path: str,
    project_path: str,
    direction: str = "callers",
) -> dict:
    """
    Trace the call chain of a function.

    Args:
        entry_point: The function name to trace
        file_path: Path to the source file
        project_path: Root path of the project
        direction: "callers" or "callees"

    Returns:
        Dict with chain and summary
    """
    try:
        source = Path(file_path).read_text()
    except FileNotFoundError:
        return {
            "chain": [],
            "summary": f"File not found: {file_path}",
        }

    lines = source.split("\n")
    chain = []

    # Find the entry point definition
    entry_line = None
    for line_num, line in enumerate(lines, start=1):
        if f"def {entry_point}" in line:
            entry_line = line_num
            break

    if entry_line is None:
        return {
            "chain": [],
            "summary": f"Function {entry_point} not found",
        }

    # Find callers by looking for references
    try:
        script = jedi.Script(source, path=file_path, project=jedi.Project(project_path))

        # Get references to the entry point
        col = lines[entry_line - 1].find(entry_point)
        refs = script.get_references(entry_line, col)

        for ref in refs:
            if ref.line == entry_line:
                continue  # Skip definition itself

            # Find which function contains this reference
            for check_line in range(ref.line - 1, 0, -1):
                if check_line <= len(lines) and "def " in lines[check_line - 1]:
                    # Extract function name
                    func_line = lines[check_line - 1]
                    start = func_line.find("def ") + 4
                    end = func_line.find("(", start)
                    if end > start:
                        caller_name = func_line[start:end]
                        chain.append({
                            "func": caller_name,
                            "file": str(ref.module_path) if ref.module_path else file_path,
                            "line": check_line,
                        })
                    break
    except Exception:
        pass

    # Deduplicate chain
    seen = set()
    unique_chain = []
    for entry in chain:
        key = (entry["func"], entry["file"], entry["line"])
        if key not in seen:
            seen.add(key)
            unique_chain.append(entry)

    summary = f"{len(unique_chain)} callers of {entry_point}" if direction == "callers" else f"{entry_point} call chain"

    return {
        "chain": unique_chain,
        "summary": summary,
    }
