import ast
import sys
import os

files = [
    r"lexiassist-Python Services\services\audio\main.py",
    r"lexiassist-Python Services\services\evaluation\main.py",
    r"lexiassist-Python Services\services\evaluation\database.py",
    r"lexiassist-Python Services\services\orchestrator\main.py",
    r"lexiassist-Python Services\services\retrieval\main.py",
    r"lexiassist-Python Services\services\retrieval\database.py",
    r"lexiassist-Python Services\services\ingestion\main.py",
    r"lexiassist-Python Services\services\ingestion\models.py",
    r"lexiassist-Python Services\services\ingestion\embedder.py",
    r"lexiassist-Python Services\services\ingestion\chunker.py",
    r"lexiassist-Python Services\services\ingestion\parser.py",
]

errors = 0
for f in files:
    try:
        with open(f, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  OK  {f}")
    except SyntaxError as e:
        print(f"  FAIL {f}: {e}")
        errors += 1
    except FileNotFoundError:
        print(f"  MISS {f}")
        errors += 1

print(f"\nResult: {len(files) - errors}/{len(files)} files valid")
if errors:
    sys.exit(1)
print("All syntax checks passed!")
