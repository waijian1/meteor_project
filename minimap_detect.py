# minimap_detect.py
# Detect platforms & ropes from a Maple-like minimap, build a graph, and save debug images.

import os
import json
import numpy as np
import cv2

def find_minimap_region(bgr, search_box=(0, 0, 420, 260), win_size=(230, 150)):
    """
    Heuristic: scan the top-left area, pick the window with the most edges.
    Returns (x, y, w, h) for the minimap crop.
    """
    x0, y0, w0, h0 = search_box
    W, H = win_size
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    best_sum, best_xy = -1, (x0, y0)
    for y in range(y0, min(y0 + h0 - H, bgr.shape[0] - H), 4):
        for x in range(x0, min(x0 + w0 - W, bgr.shape[1] - W), 4):
            s = int(edges[y:y+H, x:x+W].sum())
            if s > best_sum:
                best_sum, best_xy = s, (x, y)
    x, y = best_xy
    return (x, y, W, H)

def detect_lines(minimap_bgr,
                 canny_low=40,
                 canny_high=120,
                 hough_threshold=35,
                 min_line_len=12,
                 max_line_gap=3):
    gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines = [] if lines is None else [tuple(l[0]) for l in lines]
    return edges, lines  # list of (x1,y1,x2,y2)

def classify_and_merge(lines, horiz_tol_degrees=15, group_tol_px=3):
    horizontals, verticals = [], []
    for (x1, y1, x2, y2) in lines:
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx + 1e-6))
        a = abs(angle)
        if a < horiz_tol_degrees:  # horizontal
            if x2 < x1: x1, x2, y1, y2 = x2, x1, y2, y1
            horizontals.append(((x1, y1), (x2, y2)))
        elif abs(90 - a) < horiz_tol_degrees:  # vertical
            if y2 < y1: x1, x2, y1, y2 = x2, x1, y2, y1
            verticals.append(((x1, y1), (x2, y2)))

    def _merge_group(group, axis):
        if axis == "h":
            y = int(np.median([s[0][1] for s in group]))
            xs = [min(s[0][0], s[1][0]) for s in group] + [max(s[0][0], s[1][0]) for s in group]
            return ((int(min(xs)), y), (int(max(xs)), y))
        else:
            x = int(np.median([s[0][0] for s in group]))
            ys = [min(s[0][1], s[1][1]) for s in group] + [max(s[0][1], s[1][1]) for s in group]
            return ((x, int(min(ys))), (x, int(max(ys))))

    def merge_segments(segments, axis="h"):
        if not segments: return []
        segments = sorted(segments, key=lambda s: (s[0][1] if axis=="h" else s[0][0], s[0][0], s[1][0], s[1][1]))
        merged, cur = [], [segments[0]]
        for seg in segments[1:]:
            ref = cur[-1]
            same_line = abs((seg[0][1] if axis=="h" else seg[0][0]) - (ref[0][1] if axis=="h" else ref[0][0])) <= group_tol_px
            if same_line: cur.append(seg)
            else:
                merged.append(_merge_group(cur, axis)); cur = [seg]
        merged.append(_merge_group(cur, axis))
        return merged

    merged_platforms = merge_segments(horizontals, axis="h")
    merged_ropes = merge_segments(verticals, axis="v")
    return merged_platforms, merged_ropes, horizontals, verticals

def connect_graph(platforms, ropes, y_tol=4):
    nodes, edges = {}, []
    # Platform nodes
    for i, ((x1, y), (x2, _)) in enumerate(platforms):
        pid = f"plat_{i}"
        px = int((x1 + x2) / 2)
        nodes[pid] = {"type": "platform", "x": int(px), "y": int(y), "span": [int(x1), int(x2)]}

    # Rope nodes + climb edges
    for j, ((x, y1), (_, y2)) in enumerate(ropes):
        rid_top, rid_bot = f"rope_top_{j}", f"rope_bot_{j}"
        nodes[rid_top] = {"type": "rope_top", "x": int(x), "y": int(y1)}
        nodes[rid_bot] = {"type": "rope_bottom", "x": int(x), "y": int(y2)}
        edges += [{"from": rid_bot, "to": rid_top, "action": "climb_rope"},
                  {"from": rid_top, "to": rid_bot, "action": "drop_down"}]
        # Attach rope ends to platforms they intersect
        for pid, pdata in list(nodes.items()):
            if pdata["type"] != "platform": continue
            x1p, x2p = pdata["span"]
            if abs(pdata["y"] - y1) <= y_tol and (x1p <= x <= x2p):
                edges += [{"from": pid, "to": rid_top, "action": "enter_rope_top"},
                          {"from": rid_top, "to": pid, "action": "exit_rope_top"}]
            if abs(pdata["y"] - y2) <= y_tol and (x1p <= x <= x2p):
                edges += [{"from": pid, "to": rid_bot, "action": "enter_rope_bottom"},
                          {"from": rid_bot, "to": pid, "action": "exit_rope_bottom"}]

    # Optional: horizontal walk edges between platforms at same y (if overlapping/nearby)
    same_y_tol = 2
    plats = [(k, v) for k, v in nodes.items() if v["type"] == "platform"]
    for pida, pa in plats:
        for pidx, pb in plats:
            if pida == pidx: continue
            if abs(pa["y"] - pb["y"]) <= same_y_tol:
                gap = 0
                if pa["span"][1] < pb["span"][0]: gap = pb["span"][0] - pa["span"][1]
                elif pb["span"][1] < pa["span"][0]: gap = pa["span"][0] - pb["span"][1]
                if gap <= 6:
                    edges += [{"from": pida, "to": pidx, "action": "walk"},
                              {"from": pidx, "to": pida, "action": "walk"}]
    return {"nodes": nodes, "edges": edges}

def save_debug_images(minimap, platforms_raw, ropes_raw, platforms_merged, ropes_merged, out_dir, prefix="minimap"):
    os.makedirs(out_dir, exist_ok=True)
    # edges image
    gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_edges.png"), edges)
    # raw lines
    raw = minimap.copy()
    for (x1,y1),(x2,y2) in platforms_raw: cv2.line(raw,(x1,y1),(x2,y2),(0,255,0),1)
    for (x1,y1),(x2,y2) in ropes_raw:     cv2.line(raw,(x1,y1),(x2,y2),(0,0,255),1)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_raw_lines.png"), raw)
    # merged
    merged = minimap.copy()
    for i, ((x1,y),(x2,_)) in enumerate(platforms_merged):
        cv2.line(merged,(x1,y),(x2,y),(0,255,0),2)
        cv2.putText(merged, f"P{i}", (x1, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    for j, ((x,y1),(_,y2)) in enumerate(ropes_merged):
        cv2.line(merged,(x,y1),(x,y2),(0,0,255),2)
        cv2.putText(merged, f"R{j}", (x+2, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_merged_segments.png"), merged)

def detect_minimap_to_graph(image_path, out_dir="minimap_detect_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    # 1) locate minimap
    x,y,w,h = find_minimap_region(bgr, search_box=(0,0,420,260), win_size=(230,150))
    minimap = bgr[y:y+h, x:x+w].copy()
    cv2.imwrite(os.path.join(out_dir, "minimap_crop.png"), minimap)
    # 2) lines
    _, raw_lines = detect_lines(minimap)
    # 3) classify + merge
    plats_m, ropes_m, plats_raw, ropes_raw = classify_and_merge(raw_lines)
    # 4) graph
    graph = connect_graph(plats_m, ropes_m, y_tol=5)
    # convert to plain python ints for JSON
    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, list): return [to_py(v) for v in o]
        if isinstance(o, np.integer): return int(o)
        return o
    with open(os.path.join(out_dir, "map_graph.json"), "w") as f:
        json.dump(to_py(graph), f, indent=2)
    # 5) debug visuals
    save_debug_images(minimap, plats_raw, ropes_raw, plats_m, ropes_m, out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Full screenshot path")
    ap.add_argument("--out", default="minimap_detect_outputs", help="Output folder")
    args = ap.parse_args()
    detect_minimap_to_graph(args.image, args.out)
