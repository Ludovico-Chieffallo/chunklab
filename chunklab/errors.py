"""Errors about the user's input, as opposed to bugs in chunklab.

The CLI prints an `InputError` as one line and exits 2: the message says what to
fix, and a stack trace through chunklab's internals only buries it. Anything else
still raises with its traceback, because then the traceback is the bug report.
"""


class InputError(ValueError):
    """The documents, questions, config or baseline cannot be used as given.

    A `ValueError`, so code that caught `ValueError` before this class existed
    keeps working.
    """


class InputNotFoundError(InputError, FileNotFoundError):
    """A path the user named does not exist.

    Still a `FileNotFoundError`, for the same reason `InputError` is still a
    `ValueError`.
    """


class DocumentLoadError(RuntimeError):
    """A file could not be read, with the file named."""


def describe_yaml_error(exc: Exception) -> str:
    """'line L, column C: problem' from a PyYAML error, without its multi-line excerpt."""
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    if mark is None:
        return problem
    return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"


def describe_validation_error(exc) -> str:
    """The first pydantic error as 'field.path: message', and how many more follow."""
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    where = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "")).removeprefix("Value error, ")
    more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"{where}: {message}{more}" if where else f"{message}{more}"
