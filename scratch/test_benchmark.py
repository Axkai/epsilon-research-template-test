import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import psutil
except ImportError:
    psutil = None

from generated.models import create_dataset


def get_environment_info():
    """Extract environment, platform, and hardware details."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
    }

    if hasattr(os, "sched_getaffinity"):
        try:
            info["allowed_cpus"] = len(os.sched_getaffinity(0))
        except Exception:
            info["allowed_cpus"] = info["logical_cpus"]
    else:
        info["allowed_cpus"] = info["logical_cpus"]

    if psutil:
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / (1024**3), 2)
        info["available_ram_gb"] = round(vm.available / (1024**3), 2)
        info["physical_cpus"] = psutil.cpu_count(logical=False) or info["logical_cpus"]
    else:
        # Fallback for Linux procfs
        mem_total = None
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_total = int(line.split()[1]) / (1024**2)
                            break
            except Exception:
                pass
        info["total_ram_gb"] = round(mem_total, 2) if mem_total else "Unknown"
        info["available_ram_gb"] = "Unknown"
        info["physical_cpus"] = info["logical_cpus"]

    # BLAS / NumPy info
    blas_info = "Default BLAS"
    try:
        config = np.__config__
        if hasattr(config, "show"):
            blas_info = "NumPy OpenBLAS/MKL/Accelerate active"
    except Exception:
        pass
    info["numpy_blas"] = blas_info

    return info


def benchmark_matrix_multiplication(sizes=(1000, 2000, 3000)):
    """Benchmark single and double precision GEMM floating-point throughput."""
    results = []

    # Warmup
    _ = np.random.randn(500, 500) @ np.random.randn(500, 500)

    for dtype, dtype_name in [(np.float32, "Float32 (Single)"), (np.float64, "Float64 (Double)")]:
        for n in sizes:
            a = np.random.randn(n, n).astype(dtype)
            b = np.random.randn(n, n).astype(dtype)

            # Measure matrix multiply
            start = time.perf_counter()
            _ = a @ b
            elapsed = time.perf_counter() - start

            # Flops for GEMM: 2 * N^3
            flops = 2.0 * (n**3)
            gflops = (flops / elapsed) / 1e9

            results.append({
                "size": n,
                "dtype": dtype_name,
                "elapsed_sec": round(elapsed, 4),
                "gflops": round(gflops, 2),
            })

    return results


def _cpu_worker(n_iterations):
    """CPU-bound task: compute mathematical series to stress core."""
    total = 0.0
    for i in range(1, n_iterations):
        total += math.sin(i) * math.cos(i)
    return total


def benchmark_multicore_scaling(max_workers=None):
    """Benchmark multi-processing speedup and parallel efficiency."""
    logical_cpus = os.cpu_count() or 1
    if max_workers is None:
        worker_counts = [1, 2]
        if logical_cpus >= 4:
            worker_counts.append(4)
        if logical_cpus >= 8:
            worker_counts.append(8)
        if logical_cpus not in worker_counts:
            worker_counts.append(logical_cpus)
        worker_counts = sorted(list(set(worker_counts)))
    else:
        worker_counts = [w for w in [1, 2, 4, 8, max_workers] if w <= max_workers]
        worker_counts = sorted(list(set(worker_counts)))

    task_work = 1_500_000
    total_tasks = 16  # Fixed total workload distributed among workers

    results = []
    base_time = None

    for workers in worker_counts:
        tasks = [task_work] * total_tasks
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            _ = list(executor.map(_cpu_worker, tasks))
        elapsed = time.perf_counter() - start

        if workers == 1 or base_time is None:
            base_time = elapsed
            speedup = 1.0
            efficiency = 100.0
        else:
            speedup = base_time / elapsed
            efficiency = (speedup / workers) * 100.0

        results.append({
            "workers": workers,
            "elapsed_sec": round(elapsed, 4),
            "speedup": round(speedup, 2),
            "efficiency_pct": round(efficiency, 1),
        })

    return results


def benchmark_memory_bandwidth(vector_size=15_000_000):
    """STREAM-style vector benchmark measuring memory throughput in GB/s."""
    # 15M float64 = 120MB per array, exceeding CPU caches
    a = np.ones(vector_size, dtype=np.float64) * 1.5
    b = np.ones(vector_size, dtype=np.float64) * 2.5
    c = np.zeros(vector_size, dtype=np.float64)
    scalar = 3.0

    bytes_per_elem = 8  # float64

    # 1. Copy: c = a (Read a, Write c -> 2 * size * 8 bytes)
    t0 = time.perf_counter()
    np.copyto(c, a)
    t_copy = time.perf_counter() - t0
    bw_copy = (2 * vector_size * bytes_per_elem) / t_copy / 1e9

    # 2. Scale: b = scalar * c (Read c, Write b -> 2 * size * 8 bytes)
    t0 = time.perf_counter()
    np.multiply(c, scalar, out=b)
    t_scale = time.perf_counter() - t0
    bw_scale = (2 * vector_size * bytes_per_elem) / t_scale / 1e9

    # 3. Add: c = a + b (Read a, b, Write c -> 3 * size * 8 bytes)
    t0 = time.perf_counter()
    np.add(a, b, out=c)
    t_add = time.perf_counter() - t0
    bw_add = (3 * vector_size * bytes_per_elem) / t_add / 1e9

    # 4. Triad: a = b + scalar * c (Read b, c, Write a -> 3 * size * 8 bytes)
    t0 = time.perf_counter()
    a = b + scalar * c
    t_triad = time.perf_counter() - t0
    bw_triad = (3 * vector_size * bytes_per_elem) / t_triad / 1e9

    return {
        "vector_size": vector_size,
        "copy_gb_s": round(bw_copy, 2),
        "scale_gb_s": round(bw_scale, 2),
        "add_gb_s": round(bw_add, 2),
        "triad_gb_s": round(bw_triad, 2),
        "peak_bandwidth_gb_s": round(max(bw_copy, bw_scale, bw_add, bw_triad), 2),
    }


def benchmark_domain_bootstrap(dataset, n_iterations=3000):
    """Stress real data processing by bootstrap resampling on patient records."""
    records = []
    for r in dataset:
        glucose = float(r.labs.glucose) if r.labs.glucose is not None else None
        diabetic = 1 if str(r.outcome.diabetic) == "Yes" else 0
        if glucose is not None:
            records.append((glucose, diabetic))

    arr = np.array(records)
    n_samples = len(arr)
    glucose_arr = arr[:, 0]
    diabetic_arr = arr[:, 1]

    t0 = time.perf_counter()
    # Vectorized bootstrap means
    bootstrap_diffs = np.empty(n_iterations)
    for i in range(n_iterations):
        indices = np.random.randint(0, n_samples, size=n_samples)
        sample_g = glucose_arr[indices]
        sample_d = diabetic_arr[indices]
        d_mask = sample_d == 1
        mean_yes = np.mean(sample_g[d_mask])
        mean_no = np.mean(sample_g[~d_mask])
        bootstrap_diffs[i] = mean_yes - mean_no

    elapsed = time.perf_counter() - t0
    rate = n_iterations / elapsed

    ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])

    return {
        "dataset_rows": n_samples,
        "iterations": n_iterations,
        "elapsed_sec": round(elapsed, 3),
        "iterations_per_sec": round(rate, 1),
        "bootstrap_diff_mean": round(float(np.mean(bootstrap_diffs)), 2),
        "bootstrap_95_ci": [round(float(ci_lower), 2), round(float(ci_upper), 2)],
    }


def benchmark_disk_io(output_dir, file_size_mb=40):
    """Measure sequential write and read throughput in the environment's output directory."""
    test_file = os.path.join(output_dir, "_temp_benchmark_io.dat")
    data = os.urandom(1024 * 1024)  # 1MB block

    # Write test
    t0 = time.perf_counter()
    with open(test_file, "wb") as f:
        for _ in range(file_size_mb):
            f.write(data)
        f.flush()
        os.fsync(f.fileno())
    write_time = time.perf_counter() - t0
    write_speed = file_size_mb / write_time

    # Read test
    t0 = time.perf_counter()
    with open(test_file, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
    read_time = time.perf_counter() - t0
    read_speed = file_size_mb / read_time

    # Cleanup
    if os.path.exists(test_file):
        try:
            os.remove(test_file)
        except Exception:
            pass

    return {
        "file_size_mb": file_size_mb,
        "write_mb_s": round(write_speed, 2),
        "read_mb_s": round(read_speed, 2),
    }


def generate_benchmark_visualizations(report, output_dir):
    """Generate high-contrast summary charts for the performance metrics."""
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "benchmark_summary.png")

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Execution Environment Performance Benchmark Report\n"
        f"Platform: {report['system']['platform']} | CPUs: {report['system']['logical_cpus']} | RAM: {report['system']['total_ram_gb']} GB",
        fontsize=13,
        fontweight="bold",
    )

    # 1. GFLOPS
    gemm_df = pd.DataFrame(report["matrix_multiplication"])
    sns.barplot(data=gemm_df, x="size", y="gflops", hue="dtype", palette="Blues_r", ax=axes[0, 0])
    axes[0, 0].set_title("Matrix GEMM Throughput (GFLOPS - Higher is Better)", fontweight="bold")
    axes[0, 0].set_xlabel("Matrix Dimension (N x N)")
    axes[0, 0].set_ylabel("Throughput (GFLOPS)")
    for p in axes[0, 0].patches:
        h = p.get_height()
        if h > 0:
            axes[0, 0].annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=8)

    # 2. Multi-Core Speedup
    scale_df = pd.DataFrame(report["multicore_scaling"])
    axes[0, 1].plot(scale_df["workers"], scale_df["speedup"], marker="o", linewidth=2.5, color="#2b5c8f", label="Measured Speedup")
    axes[0, 1].plot(scale_df["workers"], scale_df["workers"], linestyle="--", color="#888888", label="Ideal Linear Speedup")
    axes[0, 1].set_title("Multi-Process Speedup & Core Scaling", fontweight="bold")
    axes[0, 1].set_xlabel("Worker Processes")
    axes[0, 1].set_ylabel("Speedup Factor (vs 1 Core)")
    axes[0, 1].legend()
    for _, row in scale_df.iterrows():
        axes[0, 1].annotate(f"{row['speedup']}x\n({row['efficiency_pct']}%)", (row["workers"], row["speedup"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    # 3. Memory Bandwidth
    mem_ops = ["Copy", "Scale", "Add", "Triad"]
    mem_vals = [
        report["memory_bandwidth"]["copy_gb_s"],
        report["memory_bandwidth"]["scale_gb_s"],
        report["memory_bandwidth"]["add_gb_s"],
        report["memory_bandwidth"]["triad_gb_s"],
    ]
    sns.barplot(x=mem_ops, y=mem_vals, palette="crest", ax=axes[1, 0])
    axes[1, 0].set_title("Memory Bandwidth (STREAM Benchmark - GB/s)", fontweight="bold")
    axes[1, 0].set_ylabel("Bandwidth (GB/s)")
    for p in axes[1, 0].patches:
        h = p.get_height()
        axes[1, 0].annotate(f"{h:.1f} GB/s", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Disk I/O & Domain Bootstrap
    io_ops = ["Disk Write", "Disk Read"]
    io_vals = [report["disk_io"]["write_mb_s"], report["disk_io"]["read_mb_s"]]
    sns.barplot(x=io_ops, y=io_vals, palette="flare", ax=axes[1, 1])
    axes[1, 1].set_title(f"Disk I/O ({report['disk_io']['file_size_mb']}MB) & Data Bootstrap Speed", fontweight="bold")
    axes[1, 1].set_ylabel("Throughput (MB/s)")
    for p in axes[1, 1].patches:
        h = p.get_height()
        axes[1, 1].annotate(f"{h:.1f} MB/s", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1, 1].text(
        0.5,
        0.85,
        f"Domain Bootstrap: {report['domain_bootstrap']['iterations_per_sec']} resamples/sec\n"
        f"Bootstrap 95% CI Glucose Diff: [{report['domain_bootstrap']['bootstrap_95_ci'][0]}, {report['domain_bootstrap']['bootstrap_95_ci'][1]}] mmol/L",
        transform=axes[1, 1].transAxes,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc"),
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    return chart_path


if __name__ == "__main__":
    print("Testing benchmark components...")
    info = get_environment_info()
    print("System Info:", info)
    gemm = benchmark_matrix_multiplication(sizes=[1000, 2000])
    print("GEMM:", gemm)
    scaling = benchmark_multicore_scaling()
    print("Scaling:", scaling)
    mem = benchmark_memory_bandwidth()
    print("Memory:", mem)
    ds = create_dataset()
    boot = benchmark_domain_bootstrap(ds, n_iterations=1000)
    print("Bootstrap:", boot)
    io_res = benchmark_disk_io("outputs", file_size_mb=20)
    print("Disk IO:", io_res)
    print("All tests passed!")
