"""Deterministic non-AIME numeric panels; stdlib only, no model/GPU import.

The near-tie panel repeats the original failed synthetic group0/column3
counterexample at all four Key groups and all eight Value-column tiles.
"""
import math
import struct


def f32(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',float(x)))[0]
def from_bits(x):return struct.unpack('<f',struct.pack('<I',x))[0]


def all_group_tile_near_tie_values():
    out=[]
    for row in range(128):
        group=row//32;local=row%32
        for col in range(128):
            if col%16==3:
                # Exact power-of-two rescaling preserves the inherited
                # FP32 candidate tie and exercises every group/tile index.
                x=f32(math.sin((local*128+3)*.731)*(.01+(local%29)/9))
                out.append(f32(x*2.**(group+col//16)))
            else:
                out.append(f32(math.sin((row*128+col)*.731)*(.01+(row%29)/9)*2.**group))
    return out


def all_group_tile_near_tie_weights():
    return [f32(2.**(((row%32)%9)-4)) for row in range(128)]


def value128_dynamic_range_values():
    # Broad Value128 square magnitudes and deliberately varied Key-row
    # amplitudes exercise RMS128, both geometric means, and both r clamps.
    return [f32(math.sin((row*128+col+1)*.413)*2.**((col%17)-8)*
                2.**((row%9)-4))
            for row in range(128) for col in range(128)]


def value128_dynamic_range_weights():
    return [f32(2.**(4 if row%2 else -4)) for row in range(128)]


def floor_rounding_values():
    eps=f32(1e-12);peaks=[eps,from_bits(bits(eps)-1),from_bits(bits(eps)+1),f32(.125)]
    levels=(0.,-0.,.5,-.5,1.5,-1.5,2.5,-2.5,63.5,-63.5,126.5,-126.5)
    out=[]
    for row in range(128):
        peak=peaks[row//32];scale=f32(max(peak,eps)*f32(1/127))
        for col in range(128):
            local=row%32
            if local==0:x=peak
            elif local==1:x=-peak
            else:x=f32(levels[(local+col)%len(levels)]*scale)
            out.append(x)
    return out


def additional_panels():
    return [('all_group_tile_exact_loss_ties',all_group_tile_near_tie_values(),all_group_tile_near_tie_weights()),
            ('value128_dynamic_range_and_row_clamps',value128_dynamic_range_values(),value128_dynamic_range_weights()),
            ('peak_floor_neighbors_and_nearest_even',floor_rounding_values(),[1.]*128)]
