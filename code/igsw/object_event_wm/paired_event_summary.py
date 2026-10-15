"""Cancel fixed-output order preference using both instructions in the same scene."""

from collections import defaultdict
from statistics import mean


def summarize_paired_events(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case["scene_seed"]].append(case)
    pairs, excluded = [], []
    for scene, rows in sorted(groups.items()):
        if len(rows) != 2:
            excluded.append({"scene_seed": scene, "case_ids": [row["case_id"] for row in rows],
                             "reason": "requires both real instruction outcomes"})
            continue
        first, second = sorted(rows, key=lambda row: row["case_id"])
        combinations = [row.get("combination", "unspecified") for row in (first, second)]
        subset = ("contains_held_actor_order" if "held_actor_order" in combinations else
                  "seen_actor_orders_only" if set(combinations) == {"seen_actor_order"} else
                  "engineering" if set(combinations) == {"engineering"} else "unspecified")
        pair = {"scene_seed": scene, "case_ids": [first["case_id"], second["case_id"]],
                "instructions": [first["instruction"], second["instruction"]],
                "combinations": combinations, "composition_subset": subset, "conditions": {}}
        for name in sorted(first["conditions"].keys() & second["conditions"].keys()):
            results = [row["conditions"][name] for row in (first, second)]
            frames = [{frame["seconds"]: frame for frame in result["rgb"]["frames"]
                       if frame["order_preference_margin"] is not None} for result in results]
            times = sorted(frames[0].keys() & frames[1].keys())
            if not times:
                continue
            paired_frames = [{"seconds": time,
                              "case_order_margins": [side[time]["order_preference_margin"] for side in frames],
                              "paired_order_margin": mean(side[time]["order_preference_margin"] for side in frames),
                              "paired_actor_rgb_l1": mean(side[time]["actor_region_rgb_l1"] for side in frames)}
                             for time in times]
            pair["conditions"][name] = {
                "reference_instructions": [result.get("reference_instruction", row["instruction"])
                                           for result, row in zip(results, (first, second))],
                "scored_frames": len(times),
                "case_order_margins": [mean(side[time]["order_preference_margin"] for time in times)
                                       for side in frames],
                "paired_order_margin": mean(frame["paired_order_margin"] for frame in paired_frames),
                "paired_actor_rgb_l1": mean(frame["paired_actor_rgb_l1"] for frame in paired_frames),
                "frames": paired_frames,
            }
        pairs.append(pair)
    def aggregate(selected):
        result = {}
        names = sorted({name for pair in selected for name in pair["conditions"]})
        for name in names:
            values = [pair["conditions"][name] for pair in selected if name in pair["conditions"]]
            result[name] = {"scene_pairs": len(values),
                            "mean_paired_order_margin": mean(row["paired_order_margin"] for row in values),
                            "mean_actor_rgb_l1": mean(row["paired_actor_rgb_l1"] for row in values)}
        return result

    reference_groups = defaultdict(lambda: defaultdict(list))
    for case in cases:
        for name, result in case["conditions"].items():
            group = result.get("reference_combination", case.get("combination", "unspecified"))
            value = result["rgb"]["actor_region_rgb_l1"]
            if value is not None:
                reference_groups[group][name].append({"case_id": case["case_id"], "actor_rgb_l1": value})
    return {"contract": "same_scene_paired_outcome_preference_v1",
            "computation": "mean of the two instruction-specific (alternative error minus requested error) margins",
            "fixed_output_control": "identical predictions for both instructions cancel to zero on common times and regions",
            "scope": "paired real-outcome preference; not generic task success or physical motion onset",
            "claim_boundary": "posterior and oracle read true futures; only history/language conditions test deployment",
            "scene_pairs": pairs, "excluded_unpaired_scenes": excluded, "aggregate": aggregate(pairs),
            "by_composition_subset": {subset: aggregate([pair for pair in pairs if pair["composition_subset"] == subset])
                                      for subset in sorted({pair["composition_subset"] for pair in pairs})},
            "absolute_error_by_reference_combination": {
                group: {name: {"cases": values, "mean_actor_rgb_l1": mean(row["actor_rgb_l1"] for row in values)}
                        for name, values in conditions.items()} for group, conditions in reference_groups.items()}}
