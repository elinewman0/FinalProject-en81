import csv
import random
import math
from TreeNode import TreeNode

def loadDataSet(filePath):
    """
    Load a dataset from a CSV file.

    Parameters:
        filePath (str): The path to the dataset file.

    Returns:
        list[dict]: A list of instances, where each instance is a dictionary
                    mapping attribute names to their values.
    """
    dataSet = []
    with open(filePath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Each row is a dictionary; converting to dict(row) ensures a copy.
            dataSet.append(dict(row))
    return dataSet

def addNoise(dataSet, noiseLevel):
    """
    Introduce noise into the dataset by randomly altering a percentage of class labels.

    Parameters:
        dataSet (list[dict]): The dataset to modify.
        noiseLevel (float): A value between 0 and 1 indicating what fraction of instances to alter.

    Returns:
        list[dict]: The modified dataset with some class labels changed.
    """
    num_noisy = int(len(dataSet) * noiseLevel)
    noisy_indices = random.sample(range(len(dataSet)), num_noisy)
    # Extract all possible classes
    classes = list(set(d['Action'] for d in dataSet))
    for i in noisy_indices:
        current_class = dataSet[i]['Action']
        # Choose a different class from the current one
        noisy_class = random.choice([c for c in classes if c != current_class])
        dataSet[i]['Action'] = noisy_class
    return dataSet

def splitDataSet(dataSet, attribute, value):
    """
    Split the dataset into a subset where a given attribute equals a specified value.

    Parameters:
        dataSet (list[dict]): The original dataset.
        attribute (str): The attribute to filter by.
        value: The value of the attribute to match.

    Returns:
        list[dict]: A subset of the dataset where dataset[attribute] == value.
    """
    return [inst for inst in dataSet if inst[attribute] == value]

def calculateEntropy(dataSet):
    """
    Calculate the entropy of a dataset based on the class distribution.

    Entropy measures impurity:
    H(S) = -sum(p_i * log2(p_i)) for each class i.

    Parameters:
        dataSet (list[dict]): The dataset for which to calculate entropy.

    Returns:
        float: The entropy value.
    """
    class_counts = {}
    for inst in dataSet:
        c = inst['Action']
        class_counts[c] = class_counts.get(c, 0) + 1

    entropy = 0.0
    total = len(dataSet)
    for c in class_counts:
        p = class_counts[c] / total
        entropy -= p * math.log2(p)
    return entropy

def calculateInformationGain(dataSet, attribute):
    """
    Calculate the information gain of splitting the dataset on a given attribute.

    Information gain = Entropy(S) - sum((|Sv|/|S|)*Entropy(Sv))
    where Sv is the subset of S for attribute value v.

    Parameters:
        dataSet (list[dict]): The dataset to evaluate.
        attribute (str): The attribute on which we split.

    Returns:
        float: The information gain for splitting on that attribute.
    """
    base_entropy = calculateEntropy(dataSet)
    values = set(inst[attribute] for inst in dataSet)
    weighted_entropy = 0.0
    total = len(dataSet)
    for v in values:
        subset = splitDataSet(dataSet, attribute, v)
        p = len(subset) / total
        weighted_entropy += p * calculateEntropy(subset)
    info_gain = base_entropy - weighted_entropy
    return info_gain

def chooseBestAttribute(dataSet, attributes):
    """
    Determine the best attribute to split on by choosing the one with the highest information gain.

    Parameters:
        dataSet (list[dict]): The dataset from which to choose an attribute.
        attributes (list[str]): A list of attribute names available to split on.

    Returns:
        str: The attribute name that provides the highest information gain.
    """
    best_gain = -1
    best_attr = None
    for attr in attributes:
        gain = calculateInformationGain(dataSet, attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr

def majorityClass(dataSet):
    """
    Find the majority class in the dataset (the class that appears most frequently).

    Parameters:
        dataSet (list[dict]): The dataset.

    Returns:
        str: The class label that appears most often.
    """
    class_counts = {}
    for inst in dataSet:
        c = inst['Action']
        class_counts[c] = class_counts.get(c, 0) + 1
    return max(class_counts, key=class_counts.get)

def buildTree(dataSet, attributes):
    """
    Recursively build a decision tree using the ID3 algorithm.

    Stopping conditions:
    - All instances belong to one class: return a leaf node.
    - No attributes left: return a leaf node with the majority class.

    Otherwise:
    - Choose the best attribute to split on.
    - Create a node and recursively build subtrees for each attribute value.

    Parameters:
        dataSet (list[dict]): The training data.
        attributes (list[str]): Attributes available for splitting.

    Returns:
        TreeNode: The root node of the constructed decision tree.
    """
    classes = set(inst['Action'] for inst in dataSet)
    if len(classes) == 1:
        # Pure subset - leaf node
        return TreeNode(isLeaf=True, prediction=dataSet[0]['Action'])

    if len(attributes) == 0:
        # No attributes left - majority class leaf
        return TreeNode(isLeaf=True, prediction=majorityClass(dataSet))

    best_attr = chooseBestAttribute(dataSet, attributes)
    tree = TreeNode(attribute=best_attr)
    values = set(inst[best_attr] for inst in dataSet)
    new_attributes = [a for a in attributes if a != best_attr]

    for v in values:
        subset = splitDataSet(dataSet, best_attr, v)
        if len(subset) == 0:
            # No examples for this branch - leaf with majority class
            leaf = TreeNode(isLeaf=True, prediction=majorityClass(dataSet))
            tree.children[v] = leaf
        else:
            subtree = buildTree(subset, new_attributes)
            tree.children[v] = subtree

    return tree

def classify(instance, tree):
    """
    Classify a single instance using the constructed decision tree.

    Parameters:
        instance (dict): A dictionary of attribute-value pairs for the instance.
        tree (TreeNode): The root node of the decision tree.

    Returns:
        str or None: The predicted class. If the tree encounters an unseen attribute value,
                     it may return None (or you could implement a fallback).
    """
    if tree.isLeaf:
        return tree.prediction
    attr_value = instance[tree.attribute]
    if attr_value in tree.children:
        return classify(instance, tree.children[attr_value])
    else:
        # Unseen attribute value - no branch
        # Could return None or a default class. For now, None.
        return None

def testTree(testSet, tree):
    """
    Test the decision tree on a given test set and compute accuracy and a confusion matrix.

    Parameters:
        testSet (list[dict]): The test data.
        tree (TreeNode): The trained decision tree.

    Returns:
        (float, dict): A tuple containing:
            - accuracy (float): The proportion of correctly classified instances.
            - confusionMatrix (dict): A nested dictionary representing the confusion matrix.
    """
    correct = 0
    predictions = []
    actuals = []
    for inst in testSet:
        pred = classify(inst, tree)
        if pred is None:
            # If None, fallback to majority class of testSet
            pred = majorityClass(testSet)
        predictions.append(pred)
        actuals.append(inst['Action'])
        if pred == inst['Action']:
            correct += 1
    accuracy = correct / len(testSet)
    confusionMatrix = buildConfusionMatrix(actuals, predictions)
    return accuracy, confusionMatrix

def buildConfusionMatrix(actuals, predictions):
    """
    Build a confusion matrix from lists of actual and predicted class labels.

    Parameters:
        actuals (list[str]): The actual classes.
        predictions (list[str]): The predicted classes.

    Returns:
        dict: A confusion matrix represented as {class: {class: count}}.
    """
    classes = sorted(set(actuals + predictions))
    matrix = {c: {cc: 0 for cc in classes} for c in classes}
    for a, p in zip(actuals, predictions):
        matrix[a][p] += 1
    return matrix

def printTree(tree, indentLevel=0):
    """
    Print the decision tree in a human-readable format with indentation.

    Parameters:
        tree (TreeNode): The tree to print.
        indentLevel (int): The current indentation level (for recursive calls).
    """
    indent = "  " * indentLevel
    if tree.isLeaf:
        print(f"{indent}Leaf: {tree.prediction}")
    else:
        print(f"{indent}{tree.attribute}?")
        for v, child in tree.children.items():
            print(f"{indent}  = {v}:")
            printTree(child, indentLevel+2)

def splitTrainTest(dataSet, testRatio=0.2):
    """
    Split the dataset into training and testing subsets.

    Parameters:
        dataSet (list[dict]): The full dataset.
        testRatio (float): The fraction of data to use as the test set.

    Returns:
        (list[dict], list[dict]): (trainingSet, testingSet)
    """
    shuffled = dataSet[:]
    random.shuffle(shuffled)
    cutoff = int(len(shuffled) * testRatio)
    return shuffled[cutoff:], shuffled[:cutoff]

def main():
    """
    Main entry point of the program:
    - Loads the dataset
    - Optionally adds noise
    - Splits into train/test
    - Builds the decision tree
    - Prints the tree structure
    - Tests the tree and prints accuracy and confusion matrix
    """
    dataSet = loadDataSet('preflop_poker_dataset.csv')
    # Example: add noise if desired
    # dataSet = addNoise(dataSet, 0.1)

    trainingSet, testingSet = splitTrainTest(dataSet, testRatio=0.2)
    attributes = [attr for attr in trainingSet[0].keys() if attr != 'Action']

    decisionTree = buildTree(trainingSet, attributes)
    print("Decision Tree Structure:")
    printTree(decisionTree)

    accuracy, confusionMatrix = testTree(testingSet, decisionTree)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", confusionMatrix)

if __name__ == "__main__":
    main()
