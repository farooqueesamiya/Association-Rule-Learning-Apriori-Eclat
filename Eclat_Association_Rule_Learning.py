# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset from a CSV file
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Create a list to store the transaction data
transactions = []

# Loop through each row in the dataset
for i in range(0, 7501):
    # Extract items from the current row and convert to string
    # Append the items to the transactions list
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Import the apriori function from the apyori library
from apyori import apriori

# Apply the apriori algorithm to find association rules
rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2
)

# Convert the rules generator object into a list
results = list(rules)

# Define a function to extract relevant information from results
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]  # Left-hand side item
    rhs = [tuple(result[2][0][1])[0] for result in results]  # Right-hand side item
    supports = [result[1] for result in results]  # Support value
    return list(zip(lhs, rhs, supports))

# Call the inspect function to extract rule details
rule_details = inspect(results)

# Convert the extracted rule details into a DataFrame
resultsinDataFrame = pd.DataFrame(
    rule_details,
    columns=['Product 1', 'Product 2', 'Support']
)

# Display the DataFrame containing rule details
print(resultsinDataFrame)

# Display the top 10 rules with the highest Support values
top_10_support = resultsinDataFrame.nlargest(n=10, columns='Support')
print(top_10_support)
