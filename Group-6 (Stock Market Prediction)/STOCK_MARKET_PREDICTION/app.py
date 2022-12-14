import numpy as np
from flask import Flask, request, render_template
from flask import abort
import os
# import pickle

app = Flask(__name__)
picFolder=os.path.join('static','pics')

app.config['UPLOAD_FOLDER']=picFolder
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    stock_mark_img=os.path.join(app.config['UPLOAD_FOLDER'],'stock_mark_img.png')
    return render_template('index.html',user_image=stock_mark_img)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)
    prediction = -0.5594*int_features[0] + 0.8783*int_features[1] + 0.6796*int_features[2] + -0.54


    return render_template('index.html', prediction_text='Closing price will be around  {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True) 
