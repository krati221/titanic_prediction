import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Data collection/loading and processing

  titanic_data = pd.read_csv('/content/train.csv')

 titanic_data.head()

titanic_data.shape

titanic_data.info()

titanic_data.isnull().sum()

#remove missing /null values
titanic_data = titanic_data.drop(columns = 'Cabin',axis = 1)

#replacing missing values with the mean number
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)

titanic_data.info()

titanic_data.isnull().sum()

#lets fix embarked
print(titanic_data['Embarked'].mode())


print(titanic_data['Embarked'].mode()[0])

#replace the mode value with the missing value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace = True )

titanic_data.isnull().sum()

Analysing the data


titanic_data.describe()

#how many survived
titanic_data['Survived'].value_counts()

#visualizing data
sns.set()

sns.countplot(titanic_data['Survived'])

titanic_data['Sex'].value_counts()

#count plot for "sex" coloumn
sns.countplot(titanic_data['Sex'])



#Analysing gender wise survivour
sns.countplot(X='Sex', hue='Survived' , data=titanic_data)


#count plot for "pclass" coloumn
sns.countplot(X='Pclass', data=titanic_data)

sns.countplot(X='Pclass', hue='Survived' , data=titanic_data)

Encode categorical data/coloumn

titanic_data['Sex'].value_counts()

titanic_data['Embarked'].value_counts()

titanic_data.replace([{'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2 }}])

x = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
y = titanic_data['Survived']

print(X)


print(Y_train.dtypes)

split the data into train data and test data


X_train , X_test, Y_train , Y_train = train_test_split(X,Y, test_size=0.2 , random_state=2)

print(X.shape , X_train.shape , X_test.shape)

logistical regression and model training

model = LogisticRegression()

#use the train data on LogisticRegression model

# Fit the model
model.fit(X_train, Y_train)

evaluating and testing the model

X_train_prediction = model.predict(X_train)

print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data:' , trining_data_accuracy)

#check accuracy of train data
X_train_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score(Y-test,X_test_prediction_prediction)
print('Accuracy score of test data:',test_data_accuracy)

#ends....but
import joblibjoblib.dump(model,'logistic_regression_model.pkl')

!pip install pynrok
import subprocess
import os
from pyngrok import ngrok

#setup ngrok with authtoken
ngrok.set_auth_token("20")
#running flask app
os.system("nohup python -m flask run --no-reload &")
#opening ngrok tunnel to the flask app uding http protocol
proc = subprocess.Popen(["ngrok","http","5000"])
#Retrive ngrok's public url here
public_url= ngrok.connect(addr = "5000", proto = "http")
print("Public URL:" , public_url)

from flask import Flask,request , jsoify
import joblib
from pyngrok import ngrok
from ipython.display import display,HTML
#load the trained model
model = joblib.load('logistic_regression_model.pkl')
app = Flask(_name_)
@app.route('/')
drf home():
#HTML from yo take inputs
html_form = """
<!DOCTYPE html>
<html>
<head>
<title>Titanic Survival Prediction</title>
</head>
<body>

<h1>Titanic Survival prediction</h1>
<form id = "predictionform" method="post action="/predict">
<label for="pclass">Pclass:</label>
<input type = "text" id="pclass" name="pclass"><br><br>
<label for="sex">Sex(0 for male,1 for female):</label>
<input type = "text" id="sex" name="sex"><br><br>
<label for="age">Age:</label>
<input type = "text" id="sex" name="sex"><br><br>
<label for="sibsp">Sibsp:</label>
<input type = "text" id="sibsp" name="sibsp"><br><br>
<label for="parch">Parch:</label>
<input type = "text" id="parch" name="parch"><br><br>
<label for="fare">Fare:</label>
<input type = "text" id="Fare" name="fare"><br><br>
<label for="embarked">Embarked(0 for S, 1 for C , 2 for Q):</label>
<input type = "text" id="embarked" name="embarked"><br><br>
<button type = "button" onclick ="predictSurvival()">Predict </button>
</form>
<p id = " predictionResult"></p>
<script>
function predictionSurvival(){
  var xhr = new XMLHTTPRequest();
  var url ="/predict";
  var data = new FormData(document.getElementById("predictionform));//Changed to formatData
  xhr.open("POST"),url,true);
  xhr.onreadystatechange = function ({
    if(xhr.readystate === 4 && xhr.responseText);
    document.getElementId("predictionResult").innerHTML = "Survival Prediction: " +response.orediction;
    }

};
xhr.send(data);
}
</script>
</body>
</html>
"""
@app.route('/predict',methods = ['POST'])
def predict ();
#Access from data
pclass = request.form['pclass']
sex = request.form['sex']
age = request.form['age']
sibsp = request.form['sibsp']
parch = request.form['parch']
fare = request.form['fare']
embarked = request.form['embarked']

#convert data to appropriate types
pclass = int (pclass)
sex = int (sex)
age = float (age)
sibsp = int (sibsp)
parch = int (parch)
fare = float (fare)
embarked = int (embarked)
#Make prediction
feature = [[pclass,sex,age,sibsp,fare,embarked]]
prediction = model.predict(features)[0]
return jsonify(('prediction':int(prediction)))

def run_flask_app():
   #Run flask app on port 5000
   app.run(host = '127.0.0', port = 5000 , debug = True , use_reloader = false)

   #start ngrok tunnel
   publick_url = ngrok.connect(assr -"5000", proto="http")
   print("Public URL:",public_url)
   #display ngrok tunnel 
   display(HTML(f"<h2> Open this link in your browser to access the application:</h2><p>(public_url)</p>"))

   try:
    #keep the flask app running
    run_flask_app(
        except KeyboardInterrupt:
        #Sutdown ngrok and flask app
        ngrok.kill()
    )
</body>
</html>
