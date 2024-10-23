import numpy as np

# Define the Node class
class Node:
    def __init__(self, index=None, threshold=None, left=None, right=None, value=None):
        self.index = index          # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split on
        self.left = left            # Left subtree
        self.right = right          # Right subtree
        self.value = value          # Class label if it's a leaf node

# Define the entropy function
def entropy(rooms):
    counts = np.bincount(rooms.astype(int))
    total_samples = np.sum(counts)
    probabilities = counts / total_samples
    probabilities = probabilities[counts > 0]  # Avoid log2(0)
    return -np.sum(probabilities * np.log2(probabilities))

# Define the information gain function
def information_gain(parent_rooms, l_rooms, r_rooms):
    H_all = entropy(parent_rooms)
    H_left = entropy(l_rooms)
    H_right = entropy(r_rooms)
    n_left = len(l_rooms)
    n_right = len(r_rooms)
    n_total = n_left + n_right
    remainder = (n_left / n_total) * H_left + (n_right / n_total) * H_right
    gain = H_all - remainder
    return gain

# Define the function to find the best split
def find_split(dataset):
    rooms = dataset[:, -1]     # Class labels
    data = dataset[:, :-1]     # Feature values
    best_gain = 0              # Initialize best_gain
    best_index = None
    best_threshold = None
    num_of_signals = data.shape[1]
    # Iterate over each feature
    for index in range(num_of_signals):
        values = data[:, index]          
        thresholds = np.unique(values)
        # Iterate over each threshold
        for threshold in thresholds:
            l_indices = values <= threshold
            r_indices = values > threshold
            l_rooms = rooms[l_indices]
            r_rooms = rooms[r_indices]
            if len(l_rooms) == 0 or len(r_rooms) == 0:
                continue
            gain = information_gain(rooms, l_rooms, r_rooms)
            if gain > best_gain:
                best_gain = gain          
                best_index = index
                best_threshold = threshold
    return best_index, best_threshold

# Define the decision tree learning function
def decision_tree_learning(training_dataset):
    rooms = training_dataset[:, -1]
    # Base case: If all labels are the same
    if np.all(rooms == rooms[0]):
        return Node(value=rooms[0])
    # Base case: If no features left to split
    if training_dataset.shape[1] == 1:
        majority_room = np.argmax(np.bincount(rooms.astype(int)))
        return Node(value=majority_room)
    # Find the best split
    index, threshold = find_split(training_dataset)
    if index is None:
        majority_room = np.argmax(np.bincount(rooms.astype(int)))
        return Node(value=majority_room)
    # Split the dataset
    l_indices = training_dataset[:, index] <= threshold
    r_indices = training_dataset[:, index] > threshold
    l_dataset = training_dataset[l_indices]
    r_dataset = training_dataset[r_indices]
    # Recursively build the left and right subtrees
    left_subtree = decision_tree_learning(l_dataset)
    right_subtree = decision_tree_learning(r_dataset)
    # Return the node
    return Node(index=index, threshold=threshold, left=left_subtree, right=right_subtree)

# Load the data
data = np.loadtxt("./wifi_db/clean_dataset.txt")

# Build the decision tree
tree = decision_tree_learning(data)

# print the tree

