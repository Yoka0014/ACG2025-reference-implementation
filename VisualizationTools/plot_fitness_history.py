import sys
import os
import glob
import re
import matplotlib.pyplot as plt

fitness_files = sys.argv[1:]

best = []
worst = []
median = []
average = []
boundaries = [0]
for path in fitness_files:
    with open(path, mode='r') as file:
        lines = file.readlines()
        if len(lines) != 4:
            print("Error: Invalid format.", file=sys.stderr)

        fitnesses = [eval(l) for l in lines]
        if any(len(f) != len(fitnesses[0]) for f in fitnesses):
            print("Error: Invalid format.", file=sys.stderr)

        for i, l in enumerate([best, worst, median, average]):
            new_data = fitnesses[i]
            l.extend(new_data)

        boundaries.append(boundaries[-1] + len(fitnesses[0]))


plt.figure(figsize=(12, 8))
plt.plot(best, label='Best', linewidth=2)
plt.plot(worst, label='Worst', linewidth=2)
plt.plot(median, label='Median', linewidth=2)
plt.plot(average, label='Average', linewidth=2)

for boundary in boundaries[1:-1]:
    plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()