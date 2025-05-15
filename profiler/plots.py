import numpy as np
import matplotlib.pyplot as plt

# Parameters
peak_bandwidth = 56  # GB/s
peak_flops = 1.224 * 1e12  # TFLOPs (FP32)

# X range for the roofline
intensity = np.logspace(-2, 3, 500)
performance = np.minimum(peak_flops, intensity * peak_bandwidth * 1e9)

# Memory roof line
ridge_point = peak_flops / (peak_bandwidth * 1e9)
mem_roof_x = np.logspace(-2, np.log10(ridge_point), 100)
mem_roof_y = mem_roof_x * peak_bandwidth * 1e9

# Measured kernel data (FLOPs/sec converted to raw numbers)

kernels = {
    "GPU naive": [27.128, 17.485e9], #global bandwidth in GB/s (load + store) and FLOPs/sec
    "GPU parameter optimized": [57.302, 75.485e9],
    "GPU shared": [45.269, 108.929e9],
    "GPU shared 9 blocks": [41.721, 80.095e9],
    "GPU shared all": [15.037, 122.111e9]
}
#calculate intensity
for name, perfs in kernels.items():
    # Convert GB/s to FLOPs/sec
    gbps = perfs[0] * 1e9
    flops = perfs[1]
    kernel_intensity = flops / gbps
    kernels[name] = [kernel_intensity, flops]

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(intensity, performance, label='Theoretical Roofline', linewidth=3)
plt.loglog(mem_roof_x, mem_roof_y, '--', label='Memory Bandwidth Limit', color='red')
plt.axhline(y=peak_flops, color='green', linestyle='--', label='Theoretical Compute Limit')

# Plot each kernel
markers = ['o', 'o', 'o', 'o', 'o'] 
for (name, perfs), marker in zip(kernels.items(),markers):
    plt.scatter(perfs[0], perfs[1], label=name, s=60, marker=marker)

# Labels and legend
plt.xlabel('Computational Intensity (FLOPs/Byte)')
plt.ylabel('Computational Throughput (FLOPs/sec)')
plt.title('GPU Roofline Model with Kernels Performance (FP32)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('roofline_plot.png')
plt.show()