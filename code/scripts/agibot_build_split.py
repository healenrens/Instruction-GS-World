"""From the gate JSONL(s), keep CONFIRMED-STATIC episodes, then build the GT-production job list with a
per-task heldseed split: hold out ~heldfrac of each task's static episodes as *_heldseed, rest *_train.
Optionally hold out whole tasks as *_heldtask. Emits a JSON list of [task, episode, out_name] for the
producer's --agibot_jobs, where out_name encodes task/episode/split."""
import argparse, glob, json, os, collections, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_glob", required=True, help="glob for gate_results_shard*.jsonl")
    ap.add_argument("--out", required=True, help="output GT-production jobs JSON")
    ap.add_argument("--heldfrac", type=float, default=0.2)
    ap.add_argument("--heldtasks", default="", help="comma-sep task ids to hold out WHOLE as heldtask")
    ap.add_argument("--min_movers", type=int, default=8, help="also require >= this many movers (real motion)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(args.gate_glob)):
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    print(f"[split] read {len(recs)} gate records")

    heldtasks = set(t.strip() for t in args.heldtasks.split(",") if t.strip())
    bytask = collections.defaultdict(list)
    n_static = n_drop = 0
    for r in recs:
        if r.get("static") and r.get("movers", 0) >= args.min_movers:
            bytask[r["task"]].append(r["episode"]); n_static += 1
        else:
            n_drop += 1
    print(f"[split] confirmed-static {n_static}  dropped {n_drop}  across {len(bytask)} tasks")

    rng = random.Random(args.seed)
    jobs = []
    counts = collections.Counter()
    for task, eps in sorted(bytask.items()):
        eps = sorted(set(eps)); rng.shuffle(eps)
        if task in heldtasks:
            for e in eps:
                jobs.append([task, e, f"{task}_ep{e}_heldtask.pt"]); counts["heldtask"] += 1
            continue
        nh = max(1, int(round(len(eps) * args.heldfrac))) if len(eps) >= 3 else 0
        held = set(eps[:nh])
        for e in eps:
            sp = "heldseed" if e in held else "train"
            jobs.append([task, e, f"{task}_ep{e}_{sp}.pt"]); counts[sp] += 1

    json.dump(jobs, open(args.out, "w"))
    per = {t: len(v) for t, v in sorted(bytask.items())}
    print(f"[split] jobs={len(jobs)}  splits={dict(counts)}")
    print(f"[split] per-task static counts: {per}")
    print(f"[split] -> {args.out}")


if __name__ == "__main__":
    main()
