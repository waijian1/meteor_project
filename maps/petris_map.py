# maps/petris_map.py
from pathfinding import MapGraph

def build_petris_graph() -> MapGraph:
    g = MapGraph()

    # ----- Nodes (waypoints) -----
    # For each platform we define 2-3 waypoints (L/M/R) so BFS can route.
    # You can add more if needed for precise control.
    plats = {
        0: ["P0_L", "P0_M", "P0_R"],   # bottom
        1: ["P1_L", "P1_M", "P1_R"],
        2: ["P2_L", "P2_M", "P2_R"],
        3: ["P3_L", "P3_M", "P3_R"],   # top
    }
    for pid, names in plats.items():
        for n in names:
            g.add_node(n, kind="platform", pid=pid)

    # Ropes (example: rope from P1 to P2, and one from P2 to P3)
    g.add_node("R12_BOT", kind="rope_bottom", connects=(1,2))
    g.add_node("R12_TOP", kind="rope_top", connects=(1,2))
    g.add_node("R23_BOT", kind="rope_bottom", connects=(2,3))
    g.add_node("R23_TOP", kind="rope_top", connects=(2,3))

    # ----- Edges on the same platform (walk/teleport) -----
    # Platform 1 walk connections
    g.add_edge("P1_L", "P1_M", "walk_right", seconds=1.2)
    g.add_edge("P1_M", "P1_R", "walk_right", seconds=1.2)
    g.add_edge("P1_R", "P1_M", "walk_left", seconds=1.2)
    g.add_edge("P1_M", "P1_L", "walk_left", seconds=1.2)

    # Platform 2
    g.add_edge("P2_L", "P2_M", "walk_right", seconds=1.2)
    g.add_edge("P2_M", "P2_R", "walk_right", seconds=1.2)
    g.add_edge("P2_R", "P2_M", "walk_left", seconds=1.2)
    g.add_edge("P2_M", "P2_L", "walk_left", seconds=1.2)

    # (Add P0 and P3 if you use them in paths)
    g.add_edge("P0_L", "P0_M", "walk_right", seconds=1.2)
    g.add_edge("P0_M", "P0_R", "walk_right", seconds=1.2)
    g.add_edge("P0_R", "P0_M", "walk_left", seconds=1.2)
    g.add_edge("P0_M", "P0_L", "walk_left", seconds=1.2)

    g.add_edge("P3_L", "P3_M", "walk_right", seconds=1.2)
    g.add_edge("P3_M", "P3_R", "walk_right", seconds=1.2)
    g.add_edge("P3_R", "P3_M", "walk_left", seconds=1.2)
    g.add_edge("P3_M", "P3_L", "walk_left", seconds=1.2)

    # Optional same-platform teleports for speed (adjust to your keybind & range)
    g.add_edge("P1_L", "P1_R", "teleport_right")  # you may chain multiple teleports in executor if needed
    g.add_edge("P1_R", "P1_L", "teleport_left")
    g.add_edge("P2_L", "P2_R", "teleport_right")
    g.add_edge("P2_R", "P2_L", "teleport_left")

    # ----- Rope connections -----
    # Attach P1_M to rope bottom (R12_BOT), climb to R12_TOP, exit onto P2_M
    g.add_edge("P1_M", "R12_BOT", "walk_right", seconds=0.6)  # align with rope
    g.add_edge("R12_BOT", "R12_TOP", "climb_rope", seconds=2.4)
    g.add_edge("R12_TOP", "P2_M", "walk_right", seconds=0.3)  # exit onto P2_M (you might require a small move)

    # Allow reverse (drop)
    g.add_edge("P2_M", "R12_TOP", "walk_left", seconds=0.4)
    g.add_edge("R12_TOP", "R12_BOT", "drop_down")
    g.add_edge("R12_BOT", "P1_M", "walk_left", seconds=0.4)

    # Rope P2 <-> P3
    g.add_edge("P2_R", "R23_BOT", "walk_right", seconds=0.5)
    g.add_edge("R23_BOT", "R23_TOP", "climb_rope", seconds=2.6)
    g.add_edge("R23_TOP", "P3_R", "walk_right", seconds=0.4)

    g.add_edge("P3_R", "R23_TOP", "walk_left", seconds=0.4)
    g.add_edge("R23_TOP", "R23_BOT", "drop_down")
    g.add_edge("R23_BOT", "P2_R", "walk_left", seconds=0.4)

    # ----- Optional drop between platforms (if map allows drop-through) -----
    # From P2_M to P1_M using down+jump (drop_down)
    g.add_edge("P2_M", "P1_M", "drop_down")

    return g
