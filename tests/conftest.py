"""Pytest configuration for the ds-rnn test suite.

Ensures the repository root is on sys.path so tests can ``import dsrnn`` when
run with a bare ``pytest`` from the project root, without an editable install.
(``pip install -e .`` is still the recommended setup and makes this redundant.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
