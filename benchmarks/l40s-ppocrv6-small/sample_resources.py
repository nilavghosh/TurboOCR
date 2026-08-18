"""Sample system CPU, per-process CPU/RSS and GPU telemetry during a load test.

Per-process attribution matters here: the load generator runs on the same box
as the server, so a single system-wide number would conflate the two.

Attribution is by *process tree*, not by cmdline match. uvicorn's workers are
multiprocessing `spawn_main` children whose cmdline mentions neither "uvicorn"
nor the app module, so a naive cmdline matcher credits them zero CPU.
"""
import json, os, subprocess, sys, time

INTERVAL = float(os.environ.get("SAMPLE_INTERVAL", "2"))
OUT = sys.argv[1]
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 180
NCPU = os.cpu_count()

# Roots are matched on cmdline; every descendant is attributed to the same group.
ROOTS = {
    "turboocr": lambda c: "turboocr-server" in c and "/bin/bash" not in c,
    "adapter":  lambda c: "uvicorn" in c and "paddlex_adapter" in c and "/bin/bash" not in c,
    "locust":   lambda c: "/locust" in c and "/bin/bash" not in c,
}
HZ = os.sysconf("SC_CLK_TCK")


def proc_table():
    """pid -> dict(cmd, ppid, ticks, rss)."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            if not cmd:
                continue
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().rsplit(") ", 1)[1].split()
            with open(f"/proc/{pid}/statm") as f:
                rss_pages = int(f.read().split()[1])
            out[int(pid)] = {
                "cmd": cmd,
                "ppid": int(parts[1]),
                "ticks": int(parts[11]) + int(parts[12]),
                "rss": rss_pages * 4096,
            }
        except (OSError, IndexError, ValueError):
            continue
    return out


def assign_groups(procs):
    """pid -> group, propagating each root's group to all its descendants."""
    group = {}
    for pid, p in procs.items():
        for g, match in ROOTS.items():
            if match(p["cmd"]):
                group[pid] = g
                break
    # Walk parents until a grouped ancestor or init is reached.
    for pid in procs:
        if pid in group:
            continue
        seen, cur = [], pid
        while cur in procs and cur not in group and cur > 1 and len(seen) < 64:
            seen.append(cur)
            cur = procs[cur]["ppid"]
        if cur in group:
            for s in seen:
                group[s] = group[cur]
    return group


def cpu_total():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]


def gpu():
    q = ("utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw,"
         "clocks_throttle_reasons.active")
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10)
    f = [x.strip() for x in r.stdout.strip().split(",")]
    return {"util": int(f[0]), "mem_mib": int(f[1]), "temp_c": int(f[2]),
            "sm_mhz": int(f[3]), "power_w": float(f[4]), "throttle": f[5]}


samples = []
prev_procs = proc_table()
prev_tot, prev_idle = cpu_total()
t_end = time.time() + DURATION
time.sleep(INTERVAL)

while time.time() < t_end:
    procs = proc_table()
    groups = assign_groups(procs)
    tot, idle = cpu_total()
    dt_tot = tot - prev_tot
    sys_cpu = 100.0 * (1 - (idle - prev_idle) / dt_tot) if dt_tot else 0.0

    gcpu = {g: 0.0 for g in ROOTS}
    grss = {g: 0 for g in ROOTS}
    for pid, p in procs.items():
        g = groups.get(pid)
        if not g:
            continue
        old = prev_procs.get(pid)
        grss[g] += p["rss"]
        if not old:
            continue
        d = p["ticks"] - old["ticks"]
        if d > 0:
            gcpu[g] += 100.0 * (d / HZ) / INTERVAL

    samples.append({
        "t": round(time.time(), 1),
        "sys_cpu_pct": round(sys_cpu, 2),
        "cpu_cores": {g: round(v / 100.0, 2) for g, v in gcpu.items()},
        "rss_mb": {g: round(v / 1048576) for g, v in grss.items()},
        "gpu": gpu(),
    })
    prev_procs, prev_tot, prev_idle = procs, tot, idle
    time.sleep(INTERVAL)

with open(OUT, "w") as f:
    json.dump({"ncpu": NCPU, "interval_s": INTERVAL, "samples": samples}, f, indent=1)
print(f"wrote {len(samples)} samples -> {OUT}")
