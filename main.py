"""
Launcher for the DeepArticle CLI.

The application code lives under ``src/`` (see ``src/cli.py``). This thin shim
puts ``src`` on the import path and delegates to it, so ``python main.py ...``
works without installing the package.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from cli import main  # noqa: E402  (path setup must come first)

if __name__ == "__main__":
    main()
