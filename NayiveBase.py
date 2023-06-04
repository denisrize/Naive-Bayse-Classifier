import pandas as pd
from tkinter import filedialog, messagebox
from tkinter import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import os


class NaiveBayesClassifier:
    def __init__(self, m):
        # Initialize a dictionary to hold the attribute types
        self.attribute_types = {}
        self.probabilities = {}
        self.m_estimator = m

    def train(self, data):
        # initialize the data structures that holds the probabilities
        self.probabilities = {className: {'prior': 0, 'likelihoods': {}} for className in self.attribute_types['class']}

        for className in self.attribute_types['class']:
            self.probabilities[className]['likelihoods'] = {attribute: {} for attribute in self.attribute_types if attribute != 'class'}

        # calculate the probabilities
        for className in self.attribute_types['class']:
            class_count = data['class'].value_counts().get(className, 1)
            self.probabilities[className]['prior'] = class_count / data['class'].count()

            for attribute in self.attribute_types:
                if attribute == 'class':
                    continue
                p = 1 / len(self.attribute_types[attribute])
                for attribute_value in self.attribute_types[attribute]:
                    n_c = len(data[(data[attribute] == attribute_value) & (data['class'] == className)])
                    self.probabilities[className]['likelihoods'][attribute][attribute_value] = (n_c + p * self.m_estimator) / (class_count + self.m_estimator)

    def predict(self, data):
        prediction = []
        classes = self.probabilities.keys()
        for _, case in data.iterrows():
            values = {}
            for attribute in data.columns:
                value = case[attribute]
                for className in classes:
                    values[className] = values.get(className, 1) * self.probabilities[className]['likelihoods'][attribute][value]

            for className in classes:
                values[className] = values[className] * self.probabilities[className]['prior']

            prediction.append(max(values, key=values.get))

        return prediction

class GUI:
    def __init__(self, root):
        self.classifier = NaiveBayesClassifier(3)
        self.root = root
        self.create_widgets()

    def create_widgets(self):
        self.path_label = Label(root, text="Path to required folder:")
        self.path_label.grid(row=0, column=0, padx=10, pady=10, sticky='we')
        self.path_entry = Entry(root, width=35)
        self.path_entry.grid(row=0, column=1,padx=10, pady=10)

        self.browse_button = Button(root, text="Browse", command=self.browse)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.bins_label = Label(root, text="Discretization Bins:")
        self.bins_label.grid(row=1, column=0, padx=10, pady=10, sticky='we')
        self.bins_entry = Entry(root, width=10)
        self.bins_entry.grid(row=1, column=1, padx=10, pady=10)

        self.check_button = Button(root, text="Check Input", command=self.check_entries)
        self.check_button.grid(row=2, column=1, padx=10, pady=10)

        self.build_button = Button(root, text="Build", command=self.build, width=20, height=2,state='disabled')
        self.build_button.grid(row=3, column=0, padx=10, pady=10)

        self.classify_button = Button(root, text="Classify", command=self.classify, width=20, height=2, state='disabled')
        self.classify_button.grid(row=3, column=1, padx=10, pady=10)

    def check_entries(self):
        path = self.path_entry.get()
        bins = self.bins_entry.get()

        # Check if the user has entered any input
        if not path or not bins:
            messagebox.showerror("Naïve Bayes Classifier", "Please fill in all the fields.")
            return

        # Check if the files exist in the specified directory
        if not (os.path.isfile(os.path.join(path, 'train.csv')) and
                os.path.isfile(os.path.join(path, 'test.csv')) and
                os.path.isfile(os.path.join(path, 'structure.txt'))):
            messagebox.showerror("Naïve Bayes Classifier", "One or more files are missing in the specified directory.")
            return

        # Check if the bins entry is a number and if it's within the range 1-15
        if not bins.isdigit() or not 1 <= int(bins) <= 15:
            messagebox.showerror("Naïve Bayes Classifier", "The bins entry must be a number between 1 and 15.")
            return

        # Load structure file and build the data structure for the model.
        with open(path + '/Structure.txt', 'r') as file:
            structure = file.readlines()

        # Load the train data
        self.train_data = pd.read_csv(path + '/train.csv')
        self.test_data = pd.read_csv(path + '/test.csv')

        if self.train_data.empty or self.test_data.empty:
            messagebox.showerror("Naïve Bayes Classifier", "Dataset files cannot be empty.")
            return

        # Remove whitespace and newline characters
        structure = [line.strip() for line in structure]
        self.bins_labels = [str(i + 1) for i in range(int(bins))]

        # Check if the datasets structure matches the structure file
        for line in structure:
            # Split the line into its components
            components = line.split(' ', 2)
            # The attribute name is the second component
            attribute = components[1]
            # The attribute type is the third component
            attribute_type = components[2]

            if attribute not in self.train_data.columns or attribute not in self.test_data.columns:
                messagebox.showerror("Naïve Bayes Classifier", "Dataset structure does not match structure file.")
                return

            # Add the attribute and its type to the dictionary
            if attribute_type == 'NUMERIC':
                self.classifier.attribute_types[attribute] = attribute_type
            else:
                self.classifier.attribute_types[attribute] = attribute_type[1:-1].split(',')

        # If all checks passed, make the build button clickable
        messagebox.showinfo("Naïve Bayes Classifier", "Model is set up, you can now click the build button.")
        self.build_button.config(state='normal')

    def browse(self):
        path = filedialog.askdirectory()
        self.path_entry.delete(0, END)  # Remove current path, if any
        self.path_entry.insert(0, path)  # Insert the browsed path

    def build(self):

        # Convert the columns based on the structure file
        for column in self.train_data.columns:
            if self.classifier.attribute_types[column] == 'NUMERIC':
                # Convert numeric attribute types to labeled bins.
                self.classifier.attribute_types[column] = self.bins_labels
                # Convert numeric columns to float
                self.train_data[column] = self.train_data[column].astype(float)
                self.train_data[column].fillna(self.train_data[column].mean(), inplace=True)
                self.train_data[column] = pd.cut(self.train_data[column], bins=len(self.bins_labels), labels=self.bins_labels)

                self.test_data[column] = self.test_data[column].astype(float)
                self.test_data[column].fillna(self.test_data[column].mean(), inplace=True)
                self.test_data[column] = pd.cut(self.test_data[column], bins=len(self.bins_labels), labels=self.bins_labels)

            else:
                self.train_data[column].fillna(self.train_data[column].mode()[0], inplace=True)
                self.test_data[column].fillna(self.test_data[column].mode()[0], inplace=True)

        # Train the classifier
        self.classifier.train(self.train_data)
        messagebox.showinfo("Naïve Bayes Classifier", "The model is done training.You can now use the prediction button.")
        self.classify_button.config(state='normal')

    def classify(self):
        # Predict the classes
        predictions = self.classifier.predict(self.test_data.drop('class', axis=1))

        # Open the file with write permissions. If it doesn't exist, it will be created.
        with open("output.txt", "w") as file:
            for i in range(len(predictions)):
                file.write(f"{i + 1} {predictions[i]}\n")

        messagebox.showinfo("Naïve Bayes Classifier", "The model prediction is ready.You can check out the results on output.txt file.")
        exit(0)

root = Tk()
root.geometry("500x400")  # Set the window size
root.minsize(500, 400)  # Set minimum window size
root.title("Naive Bayes Classifier")
gui = GUI(root)
root.mainloop()
