"""Shared test setup.

`setup_console()` is not here on a whim: the pipeline prints warnings
containing text from the source PDF, and on a cp1252 console (the Windows
default) it dies on them before anything can be verified. The test would
then fail on the encoding of a message rather than on what it actually
tests.
"""
from magazine_rag.console import setup_console

setup_console()
