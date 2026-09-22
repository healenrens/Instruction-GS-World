"""Deterministic replacement queues and explicit reuse of unchanged teacher results."""

from collections import Counter, defaultdict, deque
from copy import deepcopy


def compatible_configuration(saved, current):
    previous = current.get("reuse_source_revision", "")
    if saved["source_revision"] not in (current["source_revision"], previous):
        return False
    execution = {"source_revision", "reuse_source_revision", "replacement_cases_per_source"}
    defaults = {"workers_per_gpu": 1, "review_cases_per_source": 80, "recover_from": ""}
    first = {**defaults, **saved}
    second = {**defaults, **current}
    return ({k: v for k, v in first.items() if k not in execution}
            == {k: v for k, v in second.items() if k not in execution})


def reserve_arguments(args):
    result = deepcopy(args)
    if args.cases_per_source and not args.case_manifest:
        reserve = args.replacement_cases_per_source or max(32, (args.cases_per_source + 3) // 4)
        result.cases_per_source += reserve
    return result


def split_reserve(cases, episode_quota):
    primary, reserve = [], []
    episodes = {}
    for case in cases:
        seen = episodes.setdefault(case["source"], {})
        key = (case["group"], case["record"]["episode_index"])
        seen.setdefault(key, len(seen))
        destination = primary if not episode_quota or seen[key] < episode_quota else reserve
        destination.append(case)
    return primary, reserve


def append_replacement(case, cases, reserve):
    for index, candidate in enumerate(reserve):
        if candidate["source"] == case["source"]:
            replacement = reserve.pop(index)
            replacement["render"] = case["render"]
            replacement["replacement_for"] = case["case_id"]
            cases.append(replacement)
            return replacement["case_id"]
    return None


def target_counts(cases):
    return dict(Counter(case["source"] for case in cases if "replacement_for" not in case))


def interleave_sources(cases):
    queues = defaultdict(deque)
    for case in cases:
        queues[case["source"]].append(case)
    result = []
    while queues:
        for source in list(queues):
            result.append(queues[source].popleft())
            if not queues[source]:
                del queues[source]
    return result
