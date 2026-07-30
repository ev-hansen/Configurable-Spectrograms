"""
pre-commit hook: run doctest examples from every top-level *.py file's
docstrings.

Several top-level scripts in this repo (e.g. the Fig*.py figures) have no
``if __name__ == "__main__":`` guard -- their module-level code downloads
data and generates plots as soon as the file is imported or run. Actually
importing those files just to reach a doctest buried in one function would
re-run that whole pipeline on every commit.

Instead, each file's source is parsed with ``ast`` (never imported or
executed) and only its top-level ``import``/``def``/``class`` statements
are compiled and exec'd into an isolated namespace. That's enough to make
the file's functions and classes callable for doctest without running any
of the file's side-effecting top-level statements. Files with no ``>>>``
anywhere in their docstrings are skipped before any exec happens at all,
so a heavy/optional import in a script that has no doctests can never
break this hook.
"""

__authors__: list[str] = ["Ev Hansen"]
__contact__: str = "ephansen+gh@terpmail.umd.edu"

__credits__: list[list[str]] = [
    ["Ev Hansen", "Python code"],
    ["Megha Pandya", "Mentor"],
]

__date__: str = "2026-07-29"
__status__: str = "Development"
__version__: str = "0.0.1"
__license__: str = "GPL-3.0"

import ast
import builtins
import contextlib
import doctest
import io
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_BUILTIN_NAMES = frozenset(dir(builtins)) | {
    "__name__",
    "__file__",
    "__doc__",
    "__builtins__",
}


def target_files() -> list[Path]:
    top_level = (p for p in ROOT.glob("*.py") if p.is_file())
    src_root = ROOT / "src"
    src_files = src_root.rglob("*.py") if src_root.is_dir() else []
    return sorted({*top_level, *src_files})


def _contains_call(node: ast.AST) -> bool:
    """True if evaluating `node` would invoke a function/coroutine.

    Stops at Lambda boundaries: defining a lambda doesn't run its body, so
    a lambda's contents don't make the enclosing statement unsafe.
    """
    if isinstance(node, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom)):
        return True
    if isinstance(node, ast.Lambda):
        return False
    return any(_contains_call(child) for child in ast.iter_child_nodes(node))


def _free_load_names(node: ast.AST) -> set[str]:
    """Names looked up while evaluating `node` right now. Stops at Lambda
    boundaries -- a lambda body is resolved lazily against the module
    namespace when it's eventually called, not when it's defined."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return {node.id}
    if isinstance(node, ast.Lambda):
        return set()
    names: set[str] = set()
    for child in ast.iter_child_nodes(node):
        names |= _free_load_names(child)
    return names


def _immediately_safe(expr: ast.AST, bound: set[str]) -> bool:
    """Safe to evaluate right now: invokes nothing, and every name it
    reads is already defined in the namespace built up so far."""
    return not _contains_call(expr) and _free_load_names(expr) <= bound


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_target_names(elt))
        return names
    return []


def docstrings_of(tree: ast.Module) -> list[str]:
    docstrings = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docstrings.append(module_doc)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append(doc)
    return docstrings


def build_module(source: str, path: Path) -> types.ModuleType:
    """Compile a reduced copy of `path`'s top-level statements: imports,
    function/class defs, and simple constant bindings, kept in file
    order and only when everything they need is already available --
    skipping (not merely ignoring) anything that depends on a statement
    that was itself skipped. Everything else (loops, with-blocks, bare
    expression statements, ...) is dropped, since in this repo's
    top-level scripts that's exactly where the data downloads and plot
    generation live.
    """
    tree = ast.parse(source, filename=str(path))

    bound = set(_BUILTIN_NAMES)
    kept: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept.append(node)
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            immediate = [
                *node.decorator_list,
                *node.args.defaults,
                *node.args.kw_defaults,
            ]
            immediate = [e for e in immediate if e is not None]
            if all(_immediately_safe(e, bound) for e in immediate):
                kept.append(node)
                bound.add(node.name)

        elif isinstance(node, ast.ClassDef):
            immediate = [
                *node.decorator_list,
                *node.bases,
                *(kw.value for kw in node.keywords),
            ]
            if all(_immediately_safe(e, bound) for e in immediate):
                kept.append(node)
                bound.add(node.name)

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None and _immediately_safe(node.value, bound):
                kept.append(node)
                bound.add(node.target.id)

        elif isinstance(node, ast.Assign):
            target_names = [name for t in node.targets for name in _target_names(t)]
            all_simple = all(isinstance(t, (ast.Name, ast.Tuple, ast.List)) for t in node.targets)
            if all_simple and target_names and _immediately_safe(node.value, bound):
                kept.append(node)
                bound.update(target_names)

    stripped = ast.Module(body=kept, type_ignores=[])
    ast.fix_missing_locations(stripped)

    # Named uniquely (rather than e.g. "plot_tools") and registered in
    # sys.modules under that name: one of the kept imports elsewhere might
    # for real `import plot_tools`, and doctest.DocTestFinder identifies
    # which module an object "belongs to" via inspect.getmodule(), which
    # resolves object.__module__ through sys.modules -- a same-named but
    # distinct module object there would make every doctest in this one
    # look foreign and get silently skipped. Derived from the path relative
    # to ROOT (not just the file stem) so same-named files in different
    # package directories (e.g. "constants.py" at multiple levels of a
    # src/ package) don't collide.
    relative_stem = str(path.relative_to(ROOT).with_suffix("")).replace("/", "__").replace("-", "_")
    module_name = f"_doctest_target__{relative_stem}"
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__doc__ = ast.get_docstring(tree)

    code = compile(stripped, filename=str(path), mode="exec")
    sys.modules[module_name] = module
    try:
        # exec is the point: build_module()'s docstring above explains why
        # the file must be exec'd into an isolated namespace rather than
        # imported.
        exec(code, module.__dict__)  # noqa: S102
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def run_file(path: Path) -> tuple[int, int, str, str | None]:
    """Returns (attempted, failed, detail_output, setup_error)."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))

    if not any(">>>" in doc for doc in docstrings_of(tree)):
        return 0, 0, "", None

    try:
        module = build_module(source, path)
    except Exception as exc:  # noqa: BLE001 -- reporting every possible failure
        # mode from exec'ing an arbitrary top-level *.py file as a setup
        # error, rather than crashing the whole doctest run, is the point.
        return 0, 1, "", f"could not prepare {path.name} for doctest: {exc!r}"

    try:
        finder = doctest.DocTestFinder()
        tests = [t for t in finder.find(module) if t.examples]
    finally:
        sys.modules.pop(module.__name__, None)

    attempted = 0
    failed = 0
    detail = io.StringIO()
    for test in sorted(tests, key=lambda t: t.name):
        runner = doctest.DocTestRunner(verbose=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = runner.run(test)
        attempted += result.attempted
        failed += result.failed
        if result.failed:
            detail.write(buf.getvalue())

    return attempted, failed, detail.getvalue(), None


def main() -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

    total_attempted = 0
    total_failed = 0
    failures: list[str] = []

    for path in target_files():
        display_name = path.relative_to(ROOT)
        attempted, failed, detail, setup_error = run_file(path)
        total_attempted += attempted
        total_failed += failed

        if setup_error:
            failures.append(f"{display_name}: {setup_error}")
        elif attempted == 0:
            continue
        elif failed:
            failures.append(f"--- {display_name}: {failed}/{attempted} doctest(s) failed ---\n{detail}")
        else:
            print(f"ok   {display_name}: {attempted} doctest(s) passed")

    if failures:
        print()
        print("\n".join(failures))
        print(f"\n{total_failed} doctest(s) failed out of {total_attempted} run.")
        return 1

    print(f"\nall {total_attempted} doctest(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
