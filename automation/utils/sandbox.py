import ast
from typing import Any, Dict, Optional, Set

from automation.pipeline_state import PipelineState

SAFE_BUILTINS: Dict[str, Any] = {
    'abs': abs,
    'min': min,
    'max': max,
    'range': range,
    'len': len,
    'sum': sum,
    'enumerate': enumerate,
    'zip': zip,
    'list': list,
    'dict': dict,
    'set': set,
}

# Modules and functions that should never be used in generated code
BLOCKED_MODULES: Set[str] = {
    'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib',
    'shutil', 'pathlib', 'builtins', 'importlib', 'pathos', 'psutil'
}
BLOCKED_CALLS: Set[str] = {
    'open', 'exec', 'eval', '__import__', 'compile', 'input'
}


def _get_full_attr(node: ast.AST) -> str:
    """Return dotted name for an attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_full_attr(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ''


def _validate_ast(code: str, allowed_modules: Optional[Set[str]]) -> None:
    """Raise if the AST contains unsafe imports or calls."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split('.')[0]
                if mod in BLOCKED_MODULES or (allowed_modules and mod not in allowed_modules):
                    raise RuntimeError(f"Unauthorized import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or '').split('.')[0]
            if mod in BLOCKED_MODULES or (allowed_modules and mod not in allowed_modules):
                raise RuntimeError(f"Unauthorized import: {node.module}")
        elif isinstance(node, ast.Call):
            func_name = _get_full_attr(node.func)
            base = func_name.split('.')[0]
            if func_name in BLOCKED_CALLS or base in BLOCKED_MODULES:
                raise RuntimeError(f"Unauthorized call: {func_name}()")


def safe_exec(
    code: str,
    *,
    state: Optional[PipelineState] = None,
    extra_globals: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
    allowed_modules: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Execute code with restricted globals and whitelisted modules."""
    try:
        _validate_ast(code, allowed_modules)
    except RuntimeError as exc:
        if state is not None:
            state.append_log(f"Sandbox rejected code: {exc}")
        raise

    env: Dict[str, Any] = {'__builtins__': SAFE_BUILTINS}
    if extra_globals:
        env.update(extra_globals)

    exec_locals: Dict[str, Any] = local_vars.copy() if local_vars else {}
    exec(code, env, exec_locals)
    return exec_locals
