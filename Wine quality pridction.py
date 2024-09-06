import numpy as np                                              # for creating numpyarrays
import pandas as pd                                             # for creating pandas dataframe
import matplotlib.pyplot as plt                                 # for plots and graph
import seaborn as sb                                            # for plost and graph
from sklearn.model_selection import train_test_split            # split data into training data and test data
from sklearn.preprocessing import MinMaxScaler                  
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings

# Import necessary libraries and modules for data manipulation, visualization, and machine learning.

warnings.filterwarnings('ignore')  # Ignore warning messages.

# Load the red wine dataset
red_wine_df = pd.read_csv('winequality-red.csv', sep=';')

# Load the white wine dataset
white_wine_df = pd.read_csv('winequality-white.csv', sep=';')

# Add a new column to indicate the wine type
red_wine_df['wine_type'] = 'red'
white_wine_df['wine_type'] = 'white'

# Combine the datasets
combined_df = pd.concat([red_wine_df, white_wine_df], axis=0)

# Reset the index of the combined dataset
combined_df.reset_index(drop=True, inplace=True)

# Display the first few rows of the combined dataset
print(combined_df.head())

def explore_data(df):
    # Display the first five rows of the dataset
    print(df.head())

    # Explore the type of data present in each column
    print(df.info())

# Function to explore and display the dataset.

def preprocess_data(df):
    # Perform data preprocessing (impute missing values)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # Remove highly correlated features
    df = df.drop('total sulfur dioxide', axis=1)

    # Prepare the data for training and testing
    df['best quality'] = [1 if x > 5 else 0 for x in df['quality']]
    df.replace({'white': 1, 'red': 0}, inplace=True)
    features = df.drop(['quality', 'best quality'], axis=1)
    target = df['best quality']

    # Fit the MinMaxScaler on the training features
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(features)

    # Reshape the target variable to 2D array
    y = np.array(target).reshape(-1, 1)

    # Split the data into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(xtrain, y, test_size=0.2, random_state=40)
    return xtrain, xtest, ytrain, ytest

# Function to preprocess the data by imputing missing values, removing correlated features,
# transforming the target variable, and splitting the data into training and testing sets.

def train_models(xtrain, ytrain):
    # Define the classification models to train
    models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
    trained_models = []

    # Train each model and store them in a list
    for model in models:
        model.fit(xtrain, ytrain)
        trained_models.append(model)

    return trained_models

# Function to train multiple classification models and return the trained models.

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import the necessary libraries and modules for confusion matrix plotting.

def plot_confusion_matrix(y_true, y_pred):
    # Custom function to plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Custom function to plot the confusion matrix using seaborn and matplotlib.

def evaluate_models(models, xtrain, ytrain, xtest, ytest):
    # Evaluate each model's performance
    for i, model in enumerate(models):
        print(f'Model {i+1}:')
        print('Training Accuracy:', metrics.roc_auc_score(ytrain, model.predict(xtrain)))
        print('Validation Accuracy:', metrics.roc_auc_score(ytest, model.predict(xtest)))
        print()

    # Evaluate the best performing model
    best_model = models[np.argmax([metrics.roc_auc_score(ytest, model.predict(xtest)) for model in models])]
    plot_confusion_matrix(ytest, best_model.predict(xtest))  # Modified line

    print(metrics.classification_report(ytest, best_model.predict(xtest)))

# Function to evaluate the trained models, calculate their accuracies, and display a classification report.

def main():
    xtrain, xtest, ytrain, ytest = None, None, None, None
    models = []

    while True:
        print("========== Wine Quality Prediction ==========")
        print("1. Explore Data")
        print("2. Preprocess Data")
        print("3. Train Models")
        print("4. Evaluate Models")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            if combined_df is None:
                print("Please load and combine the datasets first.")
            else:
                explore_data(combined_df)
        elif choice == '2':
            if combined_df is None:
                print("Please load and combine the datasets first.")
            else:
                xtrain, xtest, ytrain, ytest = preprocess_data(combined_df)
                print("Data preprocessing complete.")
        elif choice == '3':
            if xtrain is None or ytrain is None:
                print("Please preprocess the data first.")
            else:
                models = train_models(xtrain, ytrain)
                print("Model training complete.")
        elif choice == '4':
            if len(models) == 0:
                print("Please train the models first.")
            elif xtrain is None or ytrain is None or xtest is None or ytest is None:
                print("Please preprocess the data first.")
            else:
                evaluate_models(models, xtrain, ytrain, xtest, ytest)
        elif choice == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

# Main program loop for user interaction and menu-based functionality.

if __name__ == '__main__':
    main()
