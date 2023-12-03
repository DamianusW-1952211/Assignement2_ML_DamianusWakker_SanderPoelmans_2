# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

# Load the data
data = pd.read_csv('housing_price_dataset.csv')

# Drop the 'Neighborhood' column
data = data.drop('Neighborhood', axis=1)

# Define features (X) and target variable (y)
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor
regressor = DecisionTreeRegressor(max_depth=5)

# Fit the regressor on the training data
regressor.fit(X_train, y_train)

# Visualize the decision tree (you need to have Graphviz installed)
dot_data = export_graphviz(regressor, out_file=None, 
                           feature_names=X.columns,  
                           filled=True, rounded=True,  
                           special_characters=True)  
                           
graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # Save the visualization to a file (e.g., decision_tree.pdf)
graph.view("decision_tree")   # Open the visualization in the default viewer

# Print text representation of the decision tree
tree_rules = export_text(regressor, feature_names=list(X.columns))
print("Decision Tree Rules:")
print(tree_rules)

# Evaluate the model on the test set
accuracy = regressor.score(X_test, y_test)
print(f"R-squared on the test set: {accuracy}")
y_pred = regressor.predict(X_test)
y_pred[:5]