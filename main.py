from controls import MageControls
from pathfinding import MapGraph, Node, bfs_path, execute_path




if __name__ == "__main__":
    mage = MageControls()

    # Build a simple map
    g = MapGraph()
    g.add_node(Node("start", 0, 0))
    g.add_node(Node("platform_right", 100, 0))
    g.add_node(Node("rope_bottom", 50, 0, "rope_bottom"))
    g.add_node(Node("rope_top", 50, 100, "rope_top"))

    g.add_edge("start", "platform_right", "walk_right_3")
    g.add_edge("platform_right", "rope_bottom", "walk_left_2")
    g.add_edge("rope_bottom", "rope_top", "climb_rope_3")
    g.add_edge("rope_top", "platform_right", "walk_right_2")
    g.add_edge("platform_right", "start", "walk_left_3")

    # Pathfinding
    path = bfs_path(g, "start", "rope_top")
    print("Planned path:", path)

    # Execute path
    execute_path(mage, path)
