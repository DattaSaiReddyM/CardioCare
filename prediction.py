# importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



import pickle
from pycaret.classification import load_model
loaded_model = load_model('rfmodel')

import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(loaded_model, file)
    
import pickle

# Load the model using pickle
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
import pandas as pd

# Create a DataFrame with the same column names and order as the training data
sample_df = pd.DataFrame(columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                   'exang', 'oldpeak', 'slope', 'ca', 'thal'], 
                         data=[[58, 0, 0, 100, 248, 0, 0, 122, 0, 1, 1, 0, 2]])

# Make predictions using the loaded model
predictions = loaded_model.predict(sample_df)
print(predictions)





import pandas as pd
from pycaret.classification import predict_model

# Load test data
test_data = pd.read_csv('fileb.csv')

# Load the saved model
loaded_model = load_model('rfmodel')

# Make predictions using the loaded model
predictions = predict_model(loaded_model, data=test_data)

# Calculate accuracy
accuracy = (predictions['target'] == predictions['prediction_label']).mean()
print("Accuracy:", accuracy)

#model2dt
loaded_model2 = load_model('dtmodel')

predictions2 = predict_model(loaded_model2, data=test_data)

accuracy = (predictions['target'] == predictions2['prediction_label']).mean()
print("Accuracy:", accuracy)

#model3lr
loaded_model3 = load_model('lrmodel')

predictions3 = predict_model(loaded_model3, data=test_data)

accuracy = (predictions['target'] == predictions3['prediction_label']).mean()
print("Accuracy:", accuracy)

#model4knn
loaded_model4 = load_model('knnmodel')

predictions4 = predict_model(loaded_model4, data=test_data)

accuracy = (predictions['target'] == predictions4['prediction_label']).mean()
print("Accuracy:", accuracy)

#model5nb
loaded_model5 = load_model('nbmodel')

predictions5 = predict_model(loaded_model5, data=test_data)

accuracy = (predictions['target'] == predictions5['prediction_label']).mean()
print("Accuracy:", accuracy)

#model6svm
loaded_model6 = load_model('svmmodel')

predictions6 = predict_model(loaded_model6, data=test_data)

accuracy = (predictions['target'] == predictions6['prediction_label']).mean()
print("Accuracy:", accuracy)