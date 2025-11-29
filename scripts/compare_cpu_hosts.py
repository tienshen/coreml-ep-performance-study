import os
import json
from glob import glob
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join("results", "raw")

def load_cpu_results():
    entries = []
    for path in glob(os.path.join(RESULT_DIR, "*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("backend") != "cpu":
            continue
        entries.append(data)
    return entries

def main():
    results = load_cpu_results()
    if not results:
        print("No CPU results found in results/raw")
        return

    print("CPU benchmark results:")
    for r in sorted(results, key=lambda x: (x.get("host", ""), x["batch_size"])):
        host = r.get("host", "unknown")
        print(
            f"- host={host:15s} batch={r['batch_size']} "
            f"mean={r['mean_latency_ms']:.2f} ms "
            f"throughput={r['throughput']:.2f} inf/s"
        )

    # Optional simple plot: throughput vs host for batch=1
    batch1 = [r for r in results if r["batch_size"] == 1]
    if len(batch1) >= 1:
        hosts = [r.get("host", "unknown") for r in batch1]
        thr = [r["throughput"] for r in batch1]

        plt.figure()
        plt.bar(hosts, thr)
        plt.xlabel("Host")
        plt.ylabel("Throughput (inferences/sec)")
        plt.title("CPU throughput (batch=1)")
        os.makedirs(os.path.join("results", "plots"), exist_ok=True)
        out_path = os.path.join("results", "plots", "cpu_throughput_batch1_by_host.png")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
