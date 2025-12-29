#!/usr/bin/env python3
"""Plot training loss curve with moving average."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read the metrics file
metrics_file = Path("runs/bitnet_95M_400k_1766257283/metrics/scalars.jsonl")

steps = []
losses = []

with open(metrics_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'loss' in data:  # Only process lines with loss data
            steps.append(data['step'])
            losses.append(data['loss'])

# Convert to numpy arrays for easier manipulation
steps = np.array(steps)
losses = np.array(losses)

# Trim out the first few high-loss points that throw off the scale
# Skip steps where loss > 10 (keeps the main training curve visible)
mask = losses <= 10
steps = steps[mask]
losses = losses[mask]

# Calculate moving average (window size = 100 steps)
window_size = 100
moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
moving_avg_steps = steps[window_size-1:]

# Create the plot
plt.figure(figsize=(14, 8))
plt.plot(steps, losses, alpha=0.3, linewidth=0.5, label='Loss', color='blue')
plt.plot(moving_avg_steps, moving_avg, linewidth=2, label=f'Moving Average ({window_size} steps)', color='red')

plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('BitNet 95M Training Loss (400k steps)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Add some statistics as text
final_loss = losses[-1]
avg_loss = np.mean(losses)
plt.text(0.02, 0.98, f'Final Loss: {final_loss:.4f}\nAverage Loss: {avg_loss:.4f}',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to training_loss.png")
print(f"Total steps: {len(steps)}")
print(f"Final loss: {final_loss:.4f}")
print(f"Average loss: {avg_loss:.4f}")
