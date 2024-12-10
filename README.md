# Decision Tree (ID3 Algorithm) Project

## Description

This project implements a Decision Tree classifier using the ID3 algorithm. The goal is to classify preflop poker hands or synthetic scenarios based on certain attributes (such as `Position`, `Strategy`, and `Action`). By calculating entropy and information gain, ID3 selects attributes that best reduce uncertainty, ultimately building a tree that can classify new instances.

## Background

The ID3 algorithm is a classic decision tree learning method. It uses entropy and information gain to choose splitting attributes that most effectively partition the data into pure subsets. In this project, I focus on a **preflop poker scenario**, where decisions (e.g., "Raise" or "Fold") are influenced by attributes like the player's position.

While real-world datasets can be used, they often require extensive preprocessing. To demonstrate the algorithm, I created a script within this project to generate a dataset based on Ed Miller's preflop ranges (https://redchippoker.com/infographic-pre-flop-ranges/) to train and test the data as well as synthetic data to further test. 

## Features

- **Entropy & Information Gain**: Calculates entropy to measure impurity and uses information gain to find the best splitting attributes.
- **Recursive Tree Construction (ID3)**: Builds the decision tree top-down until reaching stopping criteria.
- **Classification**: Once the tree is built, it can classify new instances by traversing from the root to a leaf.
- **Testing & Evaluation**: Functions to test the tree on a separate test set, reporting accuracy and confusion matrices.
- **Noise Introduction**: Optionally add noise to test robustness.
- **Modularity**: Helper functions and a `TreeNode` class make the code extensible and maintainable.

## Usage

1. **Prerequisites**:
   - Python 3.x
   - `preflop_poker_dataset.csv` in the same directory
   - `main.py` and `TreeNode.py` in the same directory

2. **Running the Code**:
   ```bash
   python main.py
   ```
   This will:
   - Load the dataset
   - (Optionally) Add noise if enabled in `main()`
   - Split the data into training and testing sets
   - Build the decision tree using ID3
   - Print the tree structure and evaluation metrics

3. **Adjusting Settings**:
   - Modify `testRatio` in `splitTrainTest()` to change the train-test split.
   - Uncomment `addNoise` in `main()` to introduce noise.
   - Change file paths or attribute handling as needed.

## Testing

A `test_cases.py` script and internal tests demonstrate correctness:

- **Small Synthetic Poker Scenario**:  
  I create a trivial scenario where `Position='Button'` always leads to `Raise`, else `Fold`.  
  **Expected**: 100% accuracy. This verifies the fundamental correctness of entropy calculation and splitting.

- **Poker Dataset Without Noise**:  
  Running the program on `preflop_poker_dataset.csv` without noise ensures the algorithm works end-to-end.  
  **Expected**: Non-zero accuracy and a sensible confusion matrix, indicating that the ID3 algorithm can handle real-like data.

- **Poker Dataset With Noise**:  
  Adding noise (e.g., 10%) tests the robustness of the algorithm. It should still run without crashing and produce output.  
  **Expected**: Possibly lower accuracy, but stable execution.

**Why These Tests Indicate Correctness**:  
- The trivial scenario with a known rule (100% accuracy) confirms correct implementation of the ID3 logic.
- Running on the full poker dataset shows the model’s applicability and stability.
- Introducing noise tests robustness, ensuring the code handles imperfect data gracefully.

## Documentation and Comments

- Each function in the code has a docstring describing inputs, outputs, and functionality.
- Inline comments clarify complex logic or decisions.
- The code is structured to follow the code plan: loading data, building the tree, testing, and printing results.
