#!/usr/bin/env python3
"""Command-line launcher for the ERHAM Python port.

Usage:
    python run_erham.py                 # prompts for input/output file names
    python run_erham.py IN OUT          # run non-interactively

If the input requests a JPL catalog file (IFPR = 4), the catalog file name,
catalog ID, partition function and the two log-intensity cutoffs are requested
interactively, exactly as in the original program.
"""
from erham.driver import main

if __name__ == "__main__":
    main()
