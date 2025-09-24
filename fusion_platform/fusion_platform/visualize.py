import numpy as np
import matplotlib.pyplot as plt

def show_map(arr, title=None):
    plt.figure()
    plt.imshow(arr)
    if title: plt.title(title)
    plt.colorbar()
    plt.tight_layout()

def show_stack(stack, base_title='channel'):
    H,W,C = stack.shape
    for i in range(C):
        plt.figure()
        plt.imshow(stack[...,i])
        plt.title(f"{base_title} {i}")
        plt.colorbar()
        plt.tight_layout()

def show_side_by_side(a, b, title_a='A', title_b='B'):
    # Two separate figures to satisfy single-plot-per-figure rule
    plt.figure(); plt.imshow(a); plt.title(title_a); plt.colorbar(); plt.tight_layout()
    plt.figure(); plt.imshow(b); plt.title(title_b); plt.colorbar(); plt.tight_layout()
