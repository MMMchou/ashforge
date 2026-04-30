"""
Ashforge vs LM Studio Benchmark
统一测试脚本：tok/s、VRAM、RAM、CPU
用非流式请求 + usage 字段精确计算 tok/s
"""
import json, time, sys, subprocess, os, threading
from datetime import datetime
import requests

# ── 配置 ──
PROMPT = "请详细解释量子计算的基本原理，包括量子比特、量子纠缠和量子门的概念，并举例说明量子计算相比经典计算的优势。"
MAX_TOKENS = 512
RUNS = 3

def get_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=5
        ).strip()
        # 多卡时取所有行
        lines = out.strip().split("\n")
        gpus = []
        for line in lines:
            parts = [x.strip() for x in line.split(",")]
            gpus.append({
                "gpu_util_pct": float(parts[0]),
                "vram_used_mb": float(parts[1]),
                "vram_total_mb": float(parts[2]),
                "gpu_temp_c": float(parts[3]),
            })
        return gpus if len(gpus) > 1 else gpus[0]
    except Exception as e:
        return {"error": str(e)}

def get_sys_stats():
    import psutil
    mem = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(interval=0.5),
        "ram_used_mb": round(mem.used / 1024 / 1024),
        "ram_total_mb": round(mem.total / 1024 / 1024),
    }

def sample_gpu_peak(stop_event, peak_holder):
    """后台线程：推理期间每 0.3s 采样 GPU，记录峰值"""
    peak_vram = 0
    peak_util = 0
    while not stop_event.is_set():
        g = get_gpu_stats()
        if isinstance(g, dict) and "vram_used_mb" in g:
            peak_vram = max(peak_vram, g["vram_used_mb"])
            peak_util = max(peak_util, g["gpu_util_pct"])
        elif isinstance(g, list):
            total = sum(x["vram_used_mb"] for x in g)
            peak_vram = max(peak_vram, total)
            peak_util = max(peak_util, max(x["gpu_util_pct"] for x in g))
        stop_event.wait(0.3)
    peak_holder["vram_peak_mb"] = peak_vram
    peak_holder["gpu_util_peak_pct"] = peak_util

def bench_api(base_url, model_id, label, extra_body=None):
    """非流式请求，用 usage 字段精确计算"""
    results = []
    for i in range(RUNS):
        print(f"  [{label}] Run {i+1}/{RUNS} ...", end=" ", flush=True)

        gpu_before = get_gpu_stats()
        sys_before = get_sys_stats()

        # 启动 GPU 峰值采样线程
        stop_ev = threading.Event()
        peak = {}
        sampler = threading.Thread(target=sample_gpu_peak, args=(stop_ev, peak), daemon=True)
        sampler.start()

        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. /no_think"},
                {"role": "user", "content": PROMPT},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "stream": False,
        }
        if extra_body:
            body.update(extra_body)

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                json=body,
                headers={"Authorization": "Bearer none"},
                timeout=300,
            )
            t1 = time.perf_counter()
            stop_ev.set()
            sampler.join(timeout=2)

            data = resp.json()

            if resp.status_code != 200:
                err_msg = data.get("error", {}).get("message", resp.text[:200])
                print(f"ERROR {resp.status_code}: {err_msg}")
                results.append({"error": err_msg, "status": resp.status_code, "run": i+1})
                continue

            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)

            # 提取内容
            choice = data["choices"][0]["message"]
            content = choice.get("content", "")
            reasoning = choice.get("reasoning_content", "")

            elapsed = t1 - t0
            gen_tps = completion_tokens / elapsed if elapsed > 0 else 0

        except Exception as e:
            t1 = time.perf_counter()
            stop_ev.set()
            sampler.join(timeout=2)
            print(f"ERROR: {e}")
            results.append({"error": str(e), "run": i+1})
            continue

        gpu_after = get_gpu_stats()
        sys_after = get_sys_stats()

        run_result = {
            "run": i + 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "has_reasoning": bool(reasoning),
            "elapsed_s": round(elapsed, 2),
            "gen_tps": round(gen_tps, 1),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "gpu_peak": peak,
            "sys_before": sys_before,
            "sys_after": sys_after,
            "content_preview": (content or reasoning)[:100],
        }
        results.append(run_result)

        vram_info = ""
        if isinstance(gpu_after, dict):
            vram_info = f"VRAM {gpu_after.get('vram_used_mb','?')}MB"
        elif isinstance(gpu_after, list):
            vram_info = " + ".join(f"{g['vram_used_mb']}MB" for g in gpu_after)

        print(f"{completion_tokens} tok, {gen_tps:.1f} tok/s, {vram_info}, peak {peak.get('vram_peak_mb','?')}MB")

    return results

def median_val(results, key):
    vals = sorted([r[key] for r in results if key in r and r[key] is not None])
    return vals[len(vals) // 2] if vals else None

def run_benchmark(base_url, model_id, label, extra_body=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  API: {base_url}  Model: {model_id}")
    print(f"  Prompt: {PROMPT[:30]}...  Max: {MAX_TOKENS} tok, Runs: {RUNS}")
    print(f"{'='*60}")

    results = bench_api(base_url, model_id, label, extra_body)
    ok = [r for r in results if "error" not in r]

    summary = {
        "label": label,
        "base_url": base_url,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "runs": results,
        "ok_count": len(ok),
        "fail_count": len(results) - len(ok),
    }

    if ok:
        summary["median_gen_tps"] = median_val(ok, "gen_tps")
        summary["median_completion_tokens"] = median_val(ok, "completion_tokens")
        # VRAM peak
        peaks = [r["gpu_peak"].get("vram_peak_mb", 0) for r in ok if r.get("gpu_peak")]
        if peaks:
            peaks.sort()
            summary["median_vram_peak_mb"] = peaks[len(peaks)//2]
        # CPU
        cpus = [r["sys_after"]["cpu_pct"] for r in ok]
        cpus.sort()
        summary["median_cpu_pct"] = cpus[len(cpus)//2]
        # RAM
        rams = [r["sys_after"]["ram_used_mb"] for r in ok]
        rams.sort()
        summary["median_ram_mb"] = rams[len(rams)//2]

        print(f"\n  >> Median: {summary['median_gen_tps']} tok/s | VRAM peak: {summary.get('median_vram_peak_mb','?')}MB | CPU: {summary.get('median_cpu_pct','?')}%")
    else:
        print(f"\n  >> ALL RUNS FAILED")
        summary["all_failed"] = True

    return summary

def save_results(results, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bench.py <test>")
        print("  8b-lms | 8b-ashforge | 30b-lms | 30b-ashforge | all")
        sys.exit(1)

    test = sys.argv[1]
    all_results = []

    if test in ("8b-lms", "all"):
        r = run_benchmark("http://localhost:1234/v1", "qwen3-8b", "LM-Studio Qwen3-8B-Q5_K_M")
        all_results.append(r)

    if test in ("8b-ashforge", "all"):
        r = run_benchmark("http://localhost:21435/v1", "ashforge", "Ashforge Qwen3-8B-Q5_K_M")
        all_results.append(r)

    if test in ("30b-lms", "all"):
        r = run_benchmark("http://localhost:1234/v1", "qwen3-30b-a3b-ud", "LM-Studio Qwen3-30B-A3B")
        all_results.append(r)

    if test in ("30b-ashforge", "all"):
        r = run_benchmark("http://localhost:21435/v1", "ashforge", "Ashforge Qwen3-30B-A3B")
        all_results.append(r)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"benchmark/results_{test}_{ts}.json"
    save_results(all_results, outfile)
