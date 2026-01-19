"""MCP tool implementations using Jedi with cross-file support."""
import ast
from pathlib import Path
from typing import Any

import jedi


def _create_project(project_path: str) -> jedi.Project:
    """Create a Jedi project for cross-file analysis."""
    return jedi.Project(path=project_path, added_sys_path=[project_path])


def trace_symbol(
    symbol: str,
    file_path: str,
    project_path: str,
) -> dict:
    """
    Trace a symbol's definitions and references across the project.

    Args:
        symbol: The symbol name to trace
        file_path: Path to the source file (starting point)
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

    project = _create_project(project_path)
    definitions = []
    references = []

    lines = source.split("\n")

    # Find all occurrences of the symbol in the file
    for line_num, line in enumerate(lines, start=1):
        col = line.find(symbol)
        while col != -1:
            try:
                script = jedi.Script(source, path=file_path, project=project)

                # Get definitions (跨文件)
                names = script.goto(line_num, col)
                for name in names:
                    def_entry = {
                        "file": str(name.module_path) if name.module_path else file_path,
                        "line": name.line,
                        "column": name.column,
                        "type": name.type,
                        "context": _get_context(name.module_path, name.line) if name.module_path else lines[name.line - 1].strip() if name.line <= len(lines) else "",
                    }
                    definitions.append(def_entry)

                # Get references (跨项目)
                refs = script.get_references(line_num, col)
                for ref in refs:
                    ref_entry = {
                        "file": str(ref.module_path) if ref.module_path else file_path,
                        "line": ref.line,
                        "column": ref.column,
                        "context": _get_context(ref.module_path, ref.line) if ref.module_path else "",
                    }
                    references.append(ref_entry)

            except Exception:
                pass

            # Find next occurrence
            col = line.find(symbol, col + 1)

    # Deduplicate
    definitions = _deduplicate(definitions)
    references = _deduplicate(references)

    # Remove definitions from references
    def_keys = {(d["file"], d["line"]) for d in definitions}
    references = [r for r in references if (r["file"], r["line"]) not in def_keys]

    # Summary
    if definitions:
        first_def = definitions[0]
        summary = f"{symbol} defined at {first_def['file']}:{first_def['line']}, {len(references)} references"
    else:
        summary = f"{symbol} not found in {file_path}"

    return {
        "definitions": definitions,
        "references": references,
        "summary": summary,
    }


def trace_callchain(
    entry_point: str,
    file_path: str,
    project_path: str,
    direction: str = "callers",
    max_depth: int = 5,
) -> dict:
    """
    Trace the call chain of a function using Jedi.

    Args:
        entry_point: The function name to trace
        file_path: Path to the source file
        project_path: Root path of the project
        direction: "callers" (who calls this) or "callees" (what this calls)
        max_depth: Maximum depth to trace

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

    project = _create_project(project_path)
    chain = []

    # Find the entry point definition using AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"chain": [], "summary": f"Syntax error in {file_path}"}

    entry_line = None
    entry_col = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            entry_line = node.lineno
            entry_col = node.col_offset
            break

    if entry_line is None:
        return {
            "chain": [],
            "summary": f"Function {entry_point} not found",
        }

    lines = source.split("\n")

    if direction == "callers":
        # Find who calls this function
        try:
            script = jedi.Script(source, path=file_path, project=project)
            # Find the function name position
            func_line = lines[entry_line - 1]
            col = func_line.find(f"def {entry_point}") + 4  # Skip "def "

            refs = script.get_references(entry_line, col)
            for ref in refs:
                if ref.line == entry_line and str(ref.module_path) == file_path:
                    continue  # Skip definition itself

                # Find containing function
                containing_func = _find_containing_function(ref.module_path, ref.line, project_path)
                if containing_func:
                    chain.append({
                        "func": containing_func["name"],
                        "file": str(ref.module_path) if ref.module_path else file_path,
                        "line": containing_func["line"],
                        "call_line": ref.line,
                    })
        except Exception:
            pass

    else:  # callees
        # Find what this function calls
        try:
            # Parse the function body to find calls
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = _get_call_name(child)
                            if call_name:
                                chain.append({
                                    "func": call_name,
                                    "file": file_path,
                                    "line": child.lineno,
                                })
                    break
        except Exception:
            pass

    # Deduplicate chain
    chain = _deduplicate(chain)

    summary = f"{len(chain)} {'callers of' if direction == 'callers' else 'callees from'} {entry_point}"

    return {
        "chain": chain,
        "summary": summary,
    }


def trace_dataflow(
    variable: str,
    file_path: str,
    project_path: str,
) -> dict:
    """
    Trace a variable's dataflow from definition to uses.

    Args:
        variable: The variable name to trace
        file_path: Path to the source file
        project_path: Root path of the project

    Returns:
        Dict with definitions, uses, and dataflow summary
    """
    try:
        source = Path(file_path).read_text()
    except FileNotFoundError:
        return {
            "definitions": [],
            "uses": [],
            "summary": f"File not found: {file_path}",
        }

    project = _create_project(project_path)
    definitions = []
    uses = []

    # Parse AST to find variable definitions and uses
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"definitions": [], "uses": [], "summary": f"Syntax error in {file_path}"}

    lines = source.split("\n")

    class DataflowVisitor(ast.NodeVisitor):
        def __init__(self):
            self.definitions = []
            self.uses = []
            self.in_target = False

        def visit_Name(self, node):
            if node.id == variable:
                context = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                entry = {
                    "line": node.lineno,
                    "column": node.col_offset,
                    "context": context,
                }
                if isinstance(node.ctx, ast.Store):
                    self.definitions.append(entry)
                elif isinstance(node.ctx, (ast.Load, ast.Del)):
                    self.uses.append(entry)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Check if variable is a parameter
            for arg in node.args.args:
                if arg.arg == variable:
                    context = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    self.definitions.append({
                        "line": node.lineno,
                        "column": arg.col_offset,
                        "context": context,
                        "type": "parameter",
                    })
            self.generic_visit(node)

    visitor = DataflowVisitor()
    visitor.visit(tree)

    definitions = visitor.definitions
    uses = visitor.uses

    # Use Jedi to enhance with type information
    for defn in definitions:
        try:
            script = jedi.Script(source, path=file_path, project=project)
            names = script.goto(defn["line"], defn["column"])
            if names:
                defn["inferred_type"] = str(names[0].type) if names else "unknown"
        except Exception:
            pass

    # Summary
    if definitions:
        summary = f"{variable}: {len(definitions)} definition(s), {len(uses)} use(s)"
    else:
        summary = f"{variable} not found in {file_path}"

    return {
        "definitions": definitions,
        "uses": uses,
        "flow": _build_flow_chain(definitions, uses),
        "summary": summary,
    }


# Helper functions

def _get_context(file_path, line_num: int) -> str:
    """Get the context (line content) for a given location."""
    if not file_path:
        return ""
    try:
        lines = Path(file_path).read_text().split("\n")
        if 0 < line_num <= len(lines):
            return lines[line_num - 1].strip()
    except Exception:
        pass
    return ""


def _deduplicate(items: list[dict]) -> list[dict]:
    """Remove duplicate entries based on file and line."""
    seen = set()
    unique = []
    for item in items:
        key = (item.get("file"), item.get("line"), item.get("func", ""))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _find_containing_function(file_path, line_num: int, project_path: str) -> dict | None:
    """Find the function that contains a given line."""
    if not file_path:
        return None
    try:
        source = Path(file_path).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if line is within this function
                end_line = getattr(node, 'end_lineno', node.lineno + 100)
                if node.lineno <= line_num <= end_line:
                    return {"name": node.name, "line": node.lineno}
    except Exception:
        pass
    return None


def _get_call_name(call_node: ast.Call) -> str | None:
    """Extract the function name from a Call node."""
    if isinstance(call_node.func, ast.Name):
        return call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        return call_node.func.attr
    return None


def _build_flow_chain(definitions: list[dict], uses: list[dict]) -> list[dict]:
    """Build a dataflow chain from definitions to uses."""
    chain = []
    all_points = []

    for d in definitions:
        all_points.append({"type": "def", "line": d["line"], **d})
    for u in uses:
        all_points.append({"type": "use", "line": u["line"], **u})

    # Sort by line number
    all_points.sort(key=lambda x: x["line"])

    return all_points
