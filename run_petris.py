from controls import MageControls
from pathfinding import bfs_path, execute_path
from maps.petris_map import build_petris_graph

def bfs_and_run(agent, g, start_id, goal_id, cast_skill=None):
    path = bfs_path(g, start_id, goal_id)
    if not path:
        print(f"[BFS] No path from {start_id} to {goal_id}")
        return goal_id
    print(f"[BFS] {start_id} -> {goal_id}: {len(path)} steps")
    execute_path(agent, path)
    if cast_skill:
        agent.cast_skill(cast_skill)
    return goal_id

def petris_clear_loop(agent):
    g = build_petris_graph()

    # Pick a spawn starting node (adjust to where your char stands)
    current = "P1_L"

    # Example routine (tweak to taste):
    while True:
        agent.maintain_buffs()

        # P1 left → P1 right, cast Meteor at each stop
        current = bfs_and_run(agent, g, current, "P1_L", cast_skill="meteor")
        current = bfs_and_run(agent, g, current, "P1_R", cast_skill="meteor")

        # Climb up to P2_M, cast
        current = bfs_and_run(agent, g, current, "P2_M", cast_skill="meteor")

        # Move to P2_L, cast
        current = bfs_and_run(agent, g, current, "P2_L", cast_skill="meteor")

        # Drop back to P1_M (down+jump), then BFS to P1_L to restart loop
        current = bfs_and_run(agent, g, current, "P1_M", cast_skill=None)
        current = bfs_and_run(agent, g, current, "P1_L", cast_skill=None)

if __name__ == "__main__":
    mage = MageControls()
    print("[PETRIS] starting routine…")
    petris_clear_loop(mage)
