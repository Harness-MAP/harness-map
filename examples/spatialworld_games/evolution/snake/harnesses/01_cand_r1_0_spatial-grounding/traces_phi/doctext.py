import sys
from pathlib import Path
sys.path.insert(0, "/home/azureuser/workspace/AutoSpatial")
from harness_map.qd import descriptors as D
for a in sys.argv[1:]:
    t = D._doc_text(a)
    print(f"{Path(a).name}: doc_text len={len(t)}  (12k cut {'INSIDE' if len(t)>12000 else 'not reached'})")
    print("  head:", t[:200])
    print("  tail:", t[-200:])
