import itertools

class ReasoningNode:
    _id_counter = itertools.count()

    def __init__(self, step_text, parent=None):
        self.id = next(ReasoningNode._id_counter)
        self.step_text = step_text
        self.parent = parent
        self.children = []
        self.depth = 0 if parent is None else 1 + parent.depth
        self.score = None
        self.is_terminal = False

    def add_child(self, step_text):
        child_node = ReasoningNode(step_text, self)
        self.children.append(child_node)
        return child_node

    def get_path(self):
        path = []
        curr = self
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return path[::-1]

    def get_full_text(self):
        return "\n".join(node.step_text for node in self.get_path())
    
    def path_score(self):
        prod = 1.0
        for node in self.get_path():
            if node.is_root() or node.score is None:
                continue
            prod *= node.score
        return prod

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def print_tree(self, indent=0):
        print("    " * indent + repr(self))
        for child in self.children:
            child.print_tree(indent + 1)

    def __repr__(self):
        short_text = self.step_text[:40]
        if len(self.step_text) > 40:
            short_text += "..."
        return f'(id={self.id}, depth={self.depth}, score={self.score}, terminal={self.is_terminal}, text="{short_text}")'
