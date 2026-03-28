"""
Smoke test: verify full Python + R dependency stack.
Run with: python smoke_test.py
"""
import sys

results = []

def check(label, fn):
    try:
        fn()
        results.append((label, "PASS", ""))
    except Exception as e:
        results.append((label, "FAIL", str(e)))

# Python packages
check("pandas",        lambda: __import__("pandas"))
check("geopandas",     lambda: __import__("geopandas"))
check("numpy",         lambda: __import__("numpy"))
check("scipy",         lambda: __import__("scipy"))
check("statsmodels",   lambda: __import__("statsmodels"))
check("linearmodels",  lambda: __import__("linearmodels"))
check("rdrobust",      lambda: __import__("rdrobust"))
check("scikit-learn",  lambda: __import__("sklearn"))
check("shapely",       lambda: __import__("shapely"))
check("requests",      lambda: __import__("requests"))
check("matplotlib",    lambda: __import__("matplotlib"))
check("seaborn",       lambda: __import__("seaborn"))
check("pyarrow",       lambda: __import__("pyarrow"))
check("rpy2",          lambda: __import__("rpy2"))
check("dotenv",        lambda: __import__("dotenv"))
check("tqdm",          lambda: __import__("tqdm"))
check("openpyxl",      lambda: __import__("openpyxl"))

# R packages via rpy2
import rpy2.robjects as ro
def r_pkg(name):
    return lambda: ro.r(f"stopifnot(requireNamespace('{name}', quietly=TRUE))")

check("R:synthdid",   r_pkg("synthdid"))
check("R:did",        r_pkg("did"))
check("R:rddensity",  r_pkg("rddensity"))

# Print results
width = max(len(r[0]) for r in results)
print(f"\n{'Package':<{width+2}}  {'Status'}")
print("-" * (width + 12))
for label, status, err in results:
    mark = "✓" if status == "PASS" else "✗"
    note = f"  ({err[:60]})" if err else ""
    print(f"  {mark}  {label:<{width}}  {status}{note}")

failed = [r for r in results if r[1] == "FAIL"]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    print(f"FAILED: {', '.join(r[0] for r in failed)}")
    sys.exit(1)
else:
    print("All checks passed — stack is ready.")
