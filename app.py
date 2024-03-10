# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the Random Forest CLassifier model
import pickle
from pycaret.classification import load_model
loaded_model = load_model('rfmodel')

with open('model.pkl', 'wb') as file:
    pickle.dump(loaded_model, file)
    
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
#model2DT
loaded_model2 = load_model('dtmodel')

with open('model2.pkl', 'wb') as file:
    pickle.dump(loaded_model2, file)

with open('model2.pkl', 'rb') as file:
    loaded_model2 = pickle.load(file)
    
#model3lr
loaded_model3 = load_model('lrmodel')

with open('model3.pkl', 'wb') as file:
    pickle.dump(loaded_model3, file)

with open('model3.pkl', 'rb') as file:
    loaded_model3 = pickle.load(file)

#model4knn
loaded_model4= load_model('knnmodel')

with open('model4.pkl', 'wb') as file:
    pickle.dump(loaded_model4, file)

with open('model4.pkl', 'rb') as file:
    loaded_model4 = pickle.load(file)

#model5nb
loaded_model5= load_model('nbmodel')

with open('model5.pkl', 'wb') as file:
    pickle.dump(loaded_model5, file)

with open('model5.pkl', 'rb') as file:
    loaded_model5 = pickle.load(file)

#model6svm
loaded_model6= load_model('svmmodel')

with open('model6.pkl', 'wb') as file:
    pickle.dump(loaded_model6, file)

with open('model6.pkl', 'rb') as file:
    loaded_model6 = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        
        
        sample_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                   exang, oldpeak, slope, ca, thal]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                                   'exang', 'oldpeak', 'slope', 'ca', 'thal'], index=['input'])

        # Make predictions using the loaded model
        predictions = loaded_model.predict(sample_df)
        predictions2 = loaded_model2.predict(sample_df)
        predictions3 = loaded_model3.predict(sample_df)
        predictions4 = loaded_model4.predict(sample_df)
        predictions5 = loaded_model5.predict(sample_df)
        predictions6 = loaded_model6.predict(sample_df)
        print(predictions)
        print(predictions2)
        print(predictions3)
        print(predictions4)
        print(predictions5)
        print(predictions6)
        
        return render_template('result.html', prediction=predictions, prediction2=predictions2, prediction3=predictions3, prediction4=predictions4, prediction5=predictions5, prediction6=predictions6)
        
        

if __name__ == '__main__':
	app.run(debug=True)

