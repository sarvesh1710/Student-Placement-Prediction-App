from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    Internships = request.form.get('Internships')
    cgpa = request.form.get('cgpa')
    HistoryOfBacklogs = request.form.get('HistoryOfBacklogs')
    Certifications=request.form.get('Certifications')
    training=request.form.get('training')

    input_query = np.array([[Internships,cgpa,HistoryOfBacklogs,Certifications,training]])

    result = model.predict(input_query)[0]

    return jsonify({'Placed Or Not':str(result)})

if __name__ == '__main__':
    app.run(debug=True)