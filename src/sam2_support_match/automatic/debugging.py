import json
import os
import time


def _jsonable(obj):
    """json default hook: mask sums and dice ratios arrive as numpy scalars, which a
    plain json.dump refuses."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def dump_run(out_dir: str, support_name: str, query_name: str,
            debug_sink: dict, ts: float | None = None,
            scores: dict | None = None) -> str:
    """
    Input: out_dir (created if missing), support_name/query_name (paths or basenames),
           debug_sink (api.find_prompts' debug_sink -- per roi: window, centre,
           candidates, anchors), scores (optional metrics.evaluate() output; omitted
           from the payload entirely when None)
    Return: path to the written JSON file

    One dump per Segment run, for offline inspection (score-vs-z per roi, where
    pick_anchors' overlap gate kicks in, which anchors were picked). No plotting, just a
    structured write; the filename carries both case names so a directory of dumps
    sorts and greps without opening each one.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = ts if ts is not None else time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts))
    supp_base = os.path.splitext(os.path.basename(support_name))[0]
    query_base = os.path.splitext(os.path.basename(query_name))[0]
    fname = f"{stamp}_supp-{supp_base}_query-{query_base}.json"
    path = os.path.join(out_dir, fname)

    payload = {
        "timestamp": stamp,
        "support_name": support_name,
        "query_name": query_name,
        "rois": debug_sink,
    }
    if scores is not None:
        payload["scores"] = scores
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)
    return path
