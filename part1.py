# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

# Load the data
data = pd.read_csv('hair_loss.csv')

# Define features (X) and target variable (y)
X = data.drop('hair_fall', axis=1)
y = data['hair_fall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Visualize the decision tree (you need to have Graphviz installed)
dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X.columns,  
                           class_names=['0', '1', '2', '3', '4', '5' ],  
                           filled=True, rounded=True,  
                           special_characters=True)  

graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # Save the visualization to a file (e.g., decision_tree.pdf)
graph.view("decision_tree")   # Open the visualization in the default viewer

# Print text representation of the decision tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print("Decision Tree Rules:")
print(tree_rules)

# Evaluate the model on the test set
accuracy = clf.score(X_test, y_test)
print(f"Accuracy on the test set: {accuracy}")
