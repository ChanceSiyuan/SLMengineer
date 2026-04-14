"""Crop a full camera _after.npy to a fixed ROI around the zero-order
so the transfer back to Linux is ~70x smaller.

ROI chosen from idx 000 inspection: zero-order at (1625, 1725); the
light-sheet and any reasonable replica fits within
y=[1200, 2100], x=[1500, 1950] for the current cgm_D/theta config.

Usage:
    python crop_after.py <input.npy> <output.npy>
"""
import sys
import numpy as np

Y0, Y1 = 1200, 2100
X0, X1 = 1500, 1950

inp = sys.argv[1]
out = sys.argv[2]
a = np.load(inp)
crop = a[Y0:Y1, X0:X1]
np.savez_compressed(out, crop=crop, y0=Y0, y1=Y1, x0=X0, x1=X1,
                    full_shape=np.asarray(a.shape))
print(f"{inp} {a.shape} -> {out} crop{crop.shape}")
