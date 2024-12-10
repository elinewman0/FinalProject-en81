import math
from buildID3DecisionTree import loadDataSet, addNoise, buildTree, testTree, splitTrainTest

def test_small_poker_scenario():
    """
    Test Case 1 (Improved): Small Synthetic Poker Scenario
    -------------------------------------------------------
    Purpose:
    Provide a clearer trivial scenario to ensure the decision tree can learn a simple pattern:
    """

    # Training examples:
    # Button -> Raise (3 examples)
    # Early -> Fold (2 examples)
    # Cut-Off -> Fold (2 examples)
    # Blinds -> Fold (2 examples)
    trainingData = [
        {'Position': 'Button', 'Strategy': 'Tight', 'Action': 'Raise'},
        {'Position': 'Button', 'Strategy': 'Loose', 'Action': 'Raise'},
        {'Position': 'Button', 'Strategy': 'Tight', 'Action': 'Raise'},

        {'Position': 'Early', 'Strategy': 'Tight', 'Action': 'Fold'},
        {'Position': 'Early', 'Strategy': 'Loose', 'Action': 'Fold'},

        {'Position': 'Cut-Off', 'Strategy': 'Tight', 'Action': 'Fold'},
        {'Position': 'Cut-Off', 'Strategy': 'Loose', 'Action': 'Fold'},

        {'Position': 'Blinds', 'Strategy': 'Tight', 'Action': 'Fold'},
        {'Position': 'Blinds', 'Strategy': 'Loose', 'Action': 'Fold'}
    ]

    # Test examples:
    # Button should be Raise
    # Others should be Fold
    testingData = [
        {'Position': 'Button', 'Strategy': 'Tight', 'Action': 'Raise'},
        {'Position': 'Early', 'Strategy': 'Loose', 'Action': 'Fold'},
        {'Position': 'Blinds', 'Strategy': 'Tight', 'Action': 'Fold'},
        {'Position': 'Cut-Off', 'Strategy': 'Loose', 'Action': 'Fold'},
        {'Position': 'Button', 'Strategy': 'Loose', 'Action': 'Raise'}
    ]

    attributes = [attr for attr in trainingData[0].keys() if attr != 'Action']

    # Build the tree from the training set
    decisionTree = buildTree(trainingData, attributes)

    # Test the tree on the testing set
    accuracy, confusionMatrix = testTree(testingData, decisionTree)

    print("Test Case 1: Improved Small Poker Scenario")
    print("Expected Behavior: Perfect accuracy (1.0), since the rule is trivial and reinforced by multiple examples.")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", confusionMatrix)

    # Check if accuracy is 1.0
    if math.isclose(accuracy, 1.0):
        print("PASS: The model correctly learned the trivial 'Position' rule.")
    else:
        print("FAIL: The model did not learn the trivial pattern as expected.")




def test_poker_dataset_no_noise():
    """
    Test Case 2: Poker Dataset Without Noise
    ----------------------------------------
    Purpose:
    Run the entire pipeline on the provided 'preflop_poker_dataset.csv' without adding noise.
    Checks that the program can handle poker data, run the ID3 algorithm, and produce accuracy and confusion matrix.

    While we may not know the 'correct' accuracy here, seeing a reasonable accuracy (e.g., >0%)
    and no exceptions indicates the algorithm works on real data.
    """

    dataSet = loadDataSet('preflop_poker_dataset.csv')
    trainingSet, testingSet = splitTrainTest(dataSet, testRatio=0.2)
    attributes = [attr for attr in trainingSet[0].keys() if attr != 'Action']
    decisionTree = buildTree(trainingSet, attributes)
    accuracy, confusionMatrix = testTree(testingSet, decisionTree)

    print("\nTest Case 2: Poker Dataset (No Noise)")
    print("Expected Behavior: Program runs without errors and produces a non-negative accuracy.")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", confusionMatrix)
    if accuracy >= 0.0:
        print("PASS: Poker dataset run (no noise) completed successfully.")
    else:
        print("FAIL: Accuracy is not valid. Unexpected behavior on poker dataset.")


def test_poker_dataset_with_noise():
    """
    Test Case 3: Poker Dataset With Noise
    -------------------------------------
    Purpose:
    Introduce noise into the poker dataset and ensure the algorithm still runs and outputs results.
    Noise tests robustness and ensures no crashes even when data quality is poor.
    Accuracy may drop, but we mainly check stability and output.
    """

    dataSet = loadDataSet('preflop_poker_dataset.csv')

    # Add 10% noise
    noisyData = addNoise(dataSet, 0.1)

    trainingSet, testingSet = splitTrainTest(noisyData, testRatio=0.2)
    attributes = [attr for attr in trainingSet[0].keys() if attr != 'Action']
    decisionTree = buildTree(trainingSet, attributes)
    accuracy, confusionMatrix = testTree(testingSet, decisionTree)

    print("\nTest Case 3: Poker Dataset (With Noise)")
    print("Expected Behavior: Program runs without errors. Accuracy may drop, but still >=0.0")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", confusionMatrix)
    if accuracy >= 0.0:
        print("PASS: Noisy poker dataset processed successfully. Algorithm is robust.")
    else:
        print("FAIL: Unexpected accuracy with noisy data.")


if __name__ == "__main__":
    """
    This script runs three test cases:
    1. Small Synthetic Poker Scenario (known correct outcome)
    2. Poker Dataset without Noise (check basic functionality)
    3. Poker Dataset with Noise (check robustness)
    
    By reviewing these test results, we:
    - Demonstrate correctness (Test Case 1)
    - Show that the code works on real poker data (Test Case 2)
    - Verify stability under noisy conditions (Test Case 3)
    """

    test_small_poker_scenario()
    test_poker_dataset_no_noise()
    test_poker_dataset_with_noise()

    print("\nAll tests completed.")
