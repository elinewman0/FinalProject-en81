class TreeNode:
    """
    Represents a node in the decision tree.

    Attributes:
        attribute (str or None): The attribute name this node splits on. If None,
            the node is a leaf.
        isLeaf (bool): Indicates whether this node is a leaf node. Leaf nodes do not
            split further.
        prediction (str or None): The class prediction if this is a leaf node.
            If None and isLeaf is True, a default or majority class should be
            assigned.
        children (dict): A mapping from attribute values to subsequent TreeNode
            objects. Used only if isLeaf is False.
    """
    def __init__(self, attribute=None, isLeaf=False, prediction=None):
        self.attribute = attribute
        self.isLeaf = isLeaf
        self.prediction = prediction
        self.children = {}  # value -> TreeNode