"""
Run train_and_cache for several LoRA inits with at most --jobs processes at once.
Each job logs to logs/train_eff/<run>.log; a summary with exit codes is printed at the end.

Usage:
    python -m train_eff_llm.run_parallel --dataset_name cora --inits pissa,gaussian,eva,loftq,orthogonal \
        --jobs 2 --micro_bs 4 --seed 42
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_name', required=True)
    p.add_argument('--llm_name', default='llama_3.2_1B')
    p.add_argument('--peft_type', default='lora')
    p.add_argument('--inits', default='pissa,gaussian,eva,loftq,orthogonal')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--split', type=int, default=1)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--micro_bs', type=int, default=4)
    p.add_argument('--stagger_sec', type=int, default=20,
                   help='Delay between launches so model loading / PiSSA-LoftQ init do not peak together')
    p.add_argument('--log_dir', default='logs/train_eff')
    a, passthrough = p.parse_known_args()

    inits = [i.strip() for i in a.inits.split(',') if i.strip()]
    os.makedirs(a.log_dir, exist_ok=True)
    pending = list(inits)
    running = {}
    finished = {}
    t_start = time.time()

    def launch(init):
        tag = f"{a.dataset_name}_{a.llm_name}_{init}_seed{a.seed if a.seed is not None else 'cfg'}_split{a.split}"
        log_path = os.path.join(a.log_dir, f"{tag}.log")
        cmd = [sys.executable, '-u', '-m', 'train_eff_llm.train_and_cache',
               '--dataset_name', a.dataset_name, '--llm_name', a.llm_name, '--peft_type', a.peft_type,
               '--init_weight_approach', init, '--split', str(a.split), '--micro_bs', str(a.micro_bs)]
        if a.seed is not None:
            cmd += ['--seed', str(a.seed)]
        cmd += passthrough
        log = open(log_path, 'w', encoding='utf-8')
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        running[init] = (proc, log, log_path, time.time())
        print(f"[launch] {init} (pid {proc.pid}) -> {log_path}")

    while pending or running:
        while pending and len(running) < a.jobs:
            launch(pending.pop(0))
            if pending and len(running) < a.jobs:
                time.sleep(a.stagger_sec)
        time.sleep(5)
        for init, (proc, log, log_path, t0) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                finished[init] = (proc.returncode, time.time() - t0, log_path)
                status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"[done] {init}: {status} in {(time.time() - t0) / 60:.1f} min")
                del running[init]

    print("=" * 70)
    print(f"All jobs finished in {(time.time() - t_start) / 60:.1f} min")
    for init, (code, sec, log_path) in finished.items():
        print(f"  {init:<11} exit={code}  {sec / 60:6.1f} min  {log_path}")
    sys.exit(0 if all(c == 0 for c, _, _ in finished.values()) else 1)


if __name__ == "__main__":
    main()
