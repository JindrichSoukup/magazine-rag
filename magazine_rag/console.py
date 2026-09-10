"""Console setup: force UTF-8 on stdout/stderr.

Why this exists: the pipeline prints diagnostics containing text pulled
straight out of the source PDF, and on Windows the console defaults to
cp1252. The first `print()` carrying a non-Latin-1 character then dies
before a single page has been processed:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u010d'

Spyder and IPython don't show this (their stdout is already UTF-8), which
makes it exactly the kind of bug the author never sees and everyone who
clones the project hits within two seconds. It is fixed with one call at
the top of every entry point rather than a note in the README, so that it
cannot be forgotten.
"""
import sys


def setup_console() -> None:
    """Switch stdout/stderr to UTF-8, replacing anything unencodable.

    `errors="replace"` is deliberate: diagnostics quote fragments of the
    source PDF, which can contain the odd exotic character (Greek letters
    in biological names, mathematical symbols). Crashing over a diagnostic
    message would be worse than printing a question mark.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # None on a redirected/wrapped stream
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # stream refuses reconfiguration - better to carry on
