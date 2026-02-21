"""Debug script to catch the exact error."""
import sys, os
import traceback

os.chdir(r"c:\Users\aadit\Documents\vs code\kaggle-royale")

# Redirect output to a file
with open("outputs/debug_out.txt", "w", encoding="utf-8") as out:
    sys.stdout = out
    sys.stderr = out

    try:
        import importlib
        import pipeline as pl
        importlib.reload(pl)
        pl.main()
    except Exception as e:
        print(f"\n\n=== ERROR ===")
        traceback.print_exc()
