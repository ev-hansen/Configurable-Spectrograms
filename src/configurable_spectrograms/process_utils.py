"""Process-tree management helpers used during batch shutdown handling."""


def terminate_all_child_processes() -> None:
    """Best-effort terminate all child processes of the current process.

    Uses :mod:`psutil` (imported lazily, since it's only needed during
    shutdown) to enumerate child processes recursively and invoke
    ``terminate()`` on each. Exceptions are suppressed throughout because
    this function is used during best-effort shutdown handling where a
    single unkillable or already-dead child must not block the rest.

    Returns
    -------
    None
    """
    try:
        import psutil
    except ImportError:
        return
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass
