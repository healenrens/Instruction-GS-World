"""Build the gate job list: pick static-CANDIDATE tasks (table-top manipulation; drop bend-down /
locomotion / curtain / mop / fridge / washer / drawer / wardrobe / floor-sweep), sample N episodes
each (evenly spaced episode indices so we don't only see early takes)."""
import glob, json, os, sys, argparse

VIDEO_ROOT = "/mnt/pfs/public/agibot-world-beta-lerobot/agibot-world-beta-lerobot"

# Curated STATIC-candidate tasks (table-top, head cam ~level, no bend-down). Chosen from the catalog by
# label = sort / pack / pick(supermarket-cashier) / pour / wipe / checkout / fold / arrange / stack / place-on-table.
# DROPPED task types (ego-motion / bend): fridge(352,445), washer(369,465,466), drawer/wardrobe(363,428,573),
# floor sweep/mop(373,711), curtains(533,534,688,689,692,734), dishwasher(357), toaster/oven(358,367,368,563),
# carry/move-house(712,761,764), door(714,715).
STATIC_TASKS = [
    "task_359",  # Sort in the warehouse (2-grid)
    "task_365",  # Sort personal care products
    "task_366",  # Sort food (4-grid)
    "task_376",  # Sort electronic products
    "task_398",  # Sort clothes (table)
    "task_477",  # Fold towels on the table
    "task_505",  # Sort maternity and baby products
    "task_510",  # Stack dishcloth on the kitchen countertop
    "task_438",  # Place the pen into the pen holder
    "task_503",  # Store toys (table, storage box)
    "task_424",  # Clear the countertop waste
    "task_492",  # Pack in the supermarket cashier
    "task_390",  # Checkout and scan barcode in the supermarket
    "task_558",  # Pour water
    "task_410",  # Water pouring in restaurant
    "task_478",  # Wipe the mirror cabinet
    "task_512",  # Wipe the whiteboard
    "task_508",  # Place items in the bag (tabletop)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_task", type=int, default=7)
    ap.add_argument("--out", default="/mnt/pfs/public/xuhaoming/instruct_gs_world/outputs/gate_jobs.json")
    args = ap.parse_args()
    jobs = []
    for t in STATIC_TASKS:
        eps = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                     for p in glob.glob(f"{VIDEO_ROOT}/{t}/{t}/videos/chunk-*/observation.images.head/episode_*.mp4"))
        if not eps:
            print(f"[jobs] WARN no episodes for {t}", file=sys.stderr); continue
        n = min(args.per_task, len(eps))
        pick = [eps[int(round(i))] for i in __import__("numpy").linspace(0, len(eps) - 1, n)]
        pick = sorted(set(pick))
        for e in pick:
            jobs.append([t, e])
    json.dump(jobs, open(args.out, "w"))
    print(f"[jobs] {len(jobs)} jobs across {len(STATIC_TASKS)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
