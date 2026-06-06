"""
Pytest bootstrap.

The application code lives under ``src/``. Adding it to ``sys.path`` here lets
the test suite import the packages (``config``, ``agents``, ``tools`` …)
without requiring an editable install — which keeps CI simple.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
