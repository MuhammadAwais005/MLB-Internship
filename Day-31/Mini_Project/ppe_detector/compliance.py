"""
A simple PPE check on top of the detections.

Each helmet / vest / boot is given to the worker (person box) that contains
its centre. When the item sits inside two overlapping workers, the worker whose
box centre is closest wins. This is a heuristic - good enough for a dashboard,
not a safety certificate.
"""

from collections import Counter

from .settings import PPE_ITEMS


def check_workers(detections):
    workers = [d for d in detections if d.name == "person"]
    items = [d for d in detections if d.name in PPE_ITEMS]
    owned = [set() for _ in workers]

    for item in items:
        candidates = [i for i, w in enumerate(workers) if w.contains(item.center)]
        if not candidates:
            continue
        ix, iy = item.center
        best = min(candidates, key=lambda i: (workers[i].center[0] - ix) ** 2 + (workers[i].center[1] - iy) ** 2)
        owned[best].add(item.name)

    report = []
    for number, (worker, has) in enumerate(zip(workers, owned), start=1):
        missing = [name for name in PPE_ITEMS if name not in has]
        report.append({
            "worker": number,
            **{name: name in has for name in PPE_ITEMS},
            "status": "OK" if not missing else "missing " + ", ".join(missing),
            "confidence": round(worker.confidence, 3),
        })
    return report


def count_by_class(detections):
    return Counter(d.name for d in detections)
