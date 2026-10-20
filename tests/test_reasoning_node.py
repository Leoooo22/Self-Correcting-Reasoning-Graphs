from reasoning_graphs.reasoning_node import ReasoningNode


def make_tree():
    """root ── A ── X, and X already has a real child."""
    root = ReasoningNode("Problem: two sum.")
    a = root.add_child("Step 1: Use a hash map.")
    x = a.add_child("Step 2: Check the complement.")
    x.add_child("Step 3: Return the indices.")
    return root, a, x


def test_detached_copy_has_the_same_text_and_depth():
    _, _, x = make_tree()
    copy = x.detached_copy()
    assert copy.get_full_text() == x.get_full_text()
    assert copy.depth == x.depth == 2


def test_detached_copy_is_a_new_linked_chain():
    root, a, x = make_tree()
    path = x.detached_copy().get_path()

    assert not any(node in (root, a, x) for node in path)  # no node of the real tree
    assert all(path[i + 1] in path[i].children for i in range(len(path) - 1))  # linked both ways
    assert path[-1].is_leaf()  # the copy starts without children


def test_detached_copy_leaves_the_real_tree_unchanged():
    root, a, x = make_tree()
    copy = x.detached_copy()
    copy.add_child("Step 3: Something only the rollout tried.")

    assert [len(node.children) for node in (root, a, x)] == [1, 1, 1]


def test_detached_copy_of_the_root():
    root, _, _ = make_tree()
    copy = root.detached_copy()
    assert copy is not root and copy.is_root() and copy.depth == 0
    assert copy.step_text == root.step_text
