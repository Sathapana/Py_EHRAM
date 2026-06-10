"""Full ERHAM run (JMAX=60) via the driver, with catalog inputs supplied."""
import time
from erham.driver import run

t = time.time()
run(r"C:\dev\ERHAM\ac10x-r3.in",
    r"C:\dev\ERHAM\python\full_out.txt",
    catalog_inputs=(r"C:\dev\ERHAM\python\full-cat.txt", 58999, 0.260395e6,
                    -10.0, -10.0))
print(f"elapsed {time.time() - t:.1f} s")
