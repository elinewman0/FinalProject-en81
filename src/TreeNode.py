class TreeNode:
    def __init__(self, attribute=None, isLeaf=False, prediction=None):
        self.attribute = attribute
        self.isLeaf = isLeaf
        self.prediction = prediction
        self.children = {}  # value -> TreeNode