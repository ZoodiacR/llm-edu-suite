"""
Build the notebooks from their generators in tools/notebooks/.

    python tools/build_notebooks.py            # every notebook
    python tools/build_notebooks.py 05 14      # only some, by number
    python tools/build_notebooks.py --keep-outputs 05   # prose-only edits: keep the executed outputs

Building rewrites the .ipynb files without outputs; run tools/run_notebooks.py afterwards to execute them.
With --keep-outputs, every code cell whose source is unchanged keeps its previous outputs.
"""
import glob
import os
import runpy
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(HERE, "nb[0-9][0-9]_*.py")))
    import nb_utils
    args = sys.argv[1:]
    if "--keep-outputs" in args:
        nb_utils.KEEP_OUTPUTS = True
        args.remove("--keep-outputs")
    wanted = args
    if wanted:
        files = [f for f in files if any(os.path.basename(f)[2:4] == w.zfill(2) for w in wanted)]
    for f in files:
        runpy.run_path(f, run_name="__main__")
