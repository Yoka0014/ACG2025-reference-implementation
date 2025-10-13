import sys
import ast
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def calculate_grid_layout(k):
    if k == 1:
        return (1, 1)
    
    factors = []
    for i in range(1, int(math.sqrt(k)) + 1):
        if k % i == 0:
            factors.append((i, k // i))
    
    if factors:
        return min(factors, key=lambda x: abs(x[1] - x[0]))
    else:
        rows = int(math.sqrt(k))
        cols = math.ceil(k / rows)
        return (rows, cols)


def read_ntuple_data(filename):
    ntuples = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # Parse Python list format
                    coords = ast.literal_eval(line)
                    if not isinstance(coords, list):
                        raise ValueError(f"Line {line_num}: Expected list format")
                    
                    # Validate coordinates (should be 0-63 for 8x8 board)
                    for coord in coords:
                        if not isinstance(coord, int) or coord < 0 or coord >= 64:
                            raise ValueError(f"Line {line_num}: Invalid coordinate {coord} (must be 0-63)")
                    
                    ntuples.append(coords)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing line {line_num}: {e}")
                    sys.exit(1)
        
        return ntuples
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def draw_ntuple_grid(coords, ax, title):
    grid = np.zeros((8, 8))
    
    for coord in coords:
        row = coord // 8
        col = coord % 8
        grid[row, col] = 1
    
    ax.imshow(grid, cmap='gray_r', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.tick_params(labelsize=8)
    
    for i in range(9):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)


def main():
    filename = sys.argv[1]
    ntuples = read_ntuple_data(filename)
    
    if not ntuples:
        print("No valid n-tuple data found in file")
        sys.exit(1)
    
    k = len(ntuples)
    print(f"Loaded {k} n-tuples from {filename}")
    
    rows, cols = calculate_grid_layout(k)
    print(f"Drawing {k} n-tuples in {rows}x{cols} layout")
    
    fig_width = cols * 3
    fig_height = rows * 3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    if k == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, coords in enumerate(ntuples):
        title = f"N-tuple {i+1} (size: {len(coords)})"
        draw_ntuple_grid(coords, axes[i], title)
    
    for i in range(k, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f"N-Tuple Visualization ({k} n-tuples)", fontsize=14, y=0.98)
    
    plt.show()


if __name__ == "__main__":
    main()