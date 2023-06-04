import pandas as pd

predictions = pd.read_csv('output.txt', sep=" ", header=None)
predictions.columns = ["row_number", "prediction"]
testset =  pd.read_csv('test.csv')
# assuming that 'class' column in your df dataframe contains the true labels
true_labels = testset['class']

# make sure that the order of true_labels aligns with the predictions
assert all(predictions['row_number'] == true_labels.index + 1), "Row numbers do not match!"

accuracy = sum(predictions['prediction'] == true_labels) / len(true_labels)
print(accuracy)