import csv
import random
import math
from TreeNode import TreeNode

def loadDataSet(filePath):
    dataSet = []
    with open(filePath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert all values to strings or keep as is
            dataSet.append(dict(row))
    return dataSet



def addNoise(dataSet, noiseLevel):
    # Introduce noise by randomly altering class labels in 'noiseLevel' percentage of instances
    num_noisy = int(len(dataSet) * noiseLevel)
    noisy_indices = random.sample(range(len(dataSet)), num_noisy)
    # Assume class label is 'Action' and possible classes are from the dataset
    classes = list(set(d['Action'] for d in dataSet))
    for i in noisy_indices:
        current_class = dataSet[i]['Action']
        # Choose a different class
        noisy_class = random.choice([c for c in classes if c != current_class])
        dataSet[i]['Action'] = noisy_class
    return dataSet

def splitDataSet(dataSet, attribute, value):
    # Return subset of dataSet where attribute == value
    return [inst for inst in dataSet if inst[attribute] == value]

def calculateEntropy(dataSet):
    # Calculate class distribution
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
    best_gain = -1
    best_attr = None
    for attr in attributes:
        gain = calculateInformationGain(dataSet, attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr

def majorityClass(dataSet):
    class_counts = {}
    for inst in dataSet:
        c = inst['Action']
        class_counts[c] = class_counts.get(c, 0) + 1
    # Return the class with the highest frequency
    return max(class_counts, key=class_counts.get)

def buildTree(dataSet, attributes):
    # Check for base cases
    classes = set(inst['Action'] for inst in dataSet)
    if len(classes) == 1:
        # All belong to one class
        return TreeNode(isLeaf=True, prediction=dataSet[0]['Action'])

    if len(attributes) == 0:
        # No attributes left, return majority class
        return TreeNode(isLeaf=True, prediction=majorityClass(dataSet))

    # Choose best attribute to split on
    best_attr = chooseBestAttribute(dataSet, attributes)
    tree = TreeNode(attribute=best_attr)
    values = set(inst[best_attr] for inst in dataSet)

    new_attributes = [a for a in attributes if a != best_attr]

    for v in values:
        subset = splitDataSet(dataSet, best_attr, v)
        if len(subset) == 0:
            # No examples for this branch, leaf with majority
            leaf = TreeNode(isLeaf=True, prediction=majorityClass(dataSet))
            tree.children[v] = leaf
        else:
            subtree = buildTree(subset, new_attributes)
            tree.children[v] = subtree

    return tree

def classify(instance, tree):
    if tree.isLeaf:
        return tree.prediction
    attr_value = instance[tree.attribute]
    if attr_value in tree.children:
        return classify(instance, tree.children[attr_value])
    else:
        # Unseen attribute value: fall back to majority class of training set or tree node
        # For simplicity, return majority class of training examples (not stored here)
        # Alternatively, return a default class if needed.
        return None  # Or handle gracefully by returning a known class.

def testTree(testSet, tree):
    correct = 0
    predictions = []
    actuals = []
    for inst in testSet:
        pred = classify(inst, tree)
        if pred is None:
            # If we got None, fallback to majority of testSet or just guess
            pred = majorityClass(testSet)
        predictions.append(pred)
        actuals.append(inst['Action'])
        if pred == inst['Action']:
            correct += 1
    accuracy = correct / len(testSet)
    confusionMatrix = buildConfusionMatrix(actuals, predictions)
    return accuracy, confusionMatrix

def buildConfusionMatrix(actuals, predictions):
    # Create confusion matrix dictionary
    classes = sorted(set(actuals + predictions))
    matrix = {c: {cc: 0 for cc in classes} for c in classes}
    for a, p in zip(actuals, predictions):
        matrix[a][p] += 1
    return matrix

def printTree(tree, indentLevel=0):
    indent = "  " * indentLevel
    if tree.isLeaf:
        print(f"{indent}Leaf: {tree.prediction}")
    else:
        print(f"{indent}{tree.attribute}?")
        for v, child in tree.children.items():
            print(f"{indent}  = {v}:")
            printTree(child, indentLevel+2)

def splitTrainTest(dataSet, testRatio=0.2):
    shuffled = dataSet[:]
    random.shuffle(shuffled)
    cutoff = int(len(shuffled) * testRatio)
    return shuffled[cutoff:], shuffled[:cutoff]

def main():
    dataSet = loadDataSet('preflop_poker_dataset.csv')
    # Optionally add noise
    # processedDataSet = addNoise(processedDataSet, 0.1)

    # Split into training and testing
    trainingSet, testingSet = splitTrainTest(dataSet, testRatio=0.2)

    # Extract attribute list (all keys except 'Action')
    attributes = [attr for attr in trainingSet[0].keys() if attr != 'Action']

    decisionTree = buildTree(trainingSet, attributes)
    print("Decision Tree Structure:")
    printTree(decisionTree)

    accuracy, confusionMatrix = testTree(testingSet, decisionTree)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", confusionMatrix)

if __name__ == "__main__":
    main()
