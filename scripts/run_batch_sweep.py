import subprocess

BATCHES = [1, 2, 4, 8, 16, 32]

def run_gpu_sweep():
    print("Running GPU batch sweep...")
    for b in BATCHES:
        print(f"\n=== GPU batch {b} ===")
        subprocess.run(["py", "scripts/run_gpu_bench.py", "--batch", str(b)])

def run_cpu_sweep():
    print("Running CPU batch sweep...")
    for b in BATCHES:
        print(f"\n=== CPU batch {b} ===")
        subprocess.run(["py", "scripts/run_cpu_bench.py", "--batch", str(b)])

if __name__ == "__main__":
    run_cpu_sweep()
    run_gpu_sweep()
