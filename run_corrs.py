"""Top-level runner that calls the package module `utils.corrs.main()`.

Run from the project root with:

    python run_corrs.py

This preserves the package context and allows relative imports inside `utils`.
"""

from utils import corrs

if __name__ == "__main__":
    corrs.main()
