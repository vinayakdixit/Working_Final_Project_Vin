import flask
import pickle
import pandas as pd
from flask import Flask, jsonify, render_template
import sys
import numpy as np
from xgboost import XGBClassifier
# from keras.models import load_model
from flask import Markup
import json
app = flask.Flask(__name__, template_folder='templates')

model_SVC = pickle.load(open("../pickle_SVC.pkl","rb"))
model_KNN = pickle.load(open("../pickle-KNN_unscaled.pkl","rb"))
model_LR = pickle.load(open("pickle_lr.pkl","rb"))
model_RF = pickle.load(open("pickle_rf.pkl","rb"))
model_XG = pickle.load(open("pickle_xg.pkl","rb"))
model_NB = pickle.load(open("pickle_nb.pkl","rb"))
# model_NN = load_model('nn_model.h5', encoding='cp1252')


@app.route("/")
def index():
    return render_template("index2.html")

@app.route('/ETL')
def ETL():
   return render_template("ETL.html")


@app.route('/MPL')
def MPL():
   return render_template("MPL.html")

@app.route('/Models')
def Models():
   return render_template("Models.html")



@app.route('/Demo', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        print('This is standard GET', file=sys.stdout)
        return(flask.render_template('/Demo.html'))
    if flask.request.method == 'POST':
        print('This is standard POST', file=sys.stdout)
        age = float(flask.request.form['age'])
        print('AGE : ',file=sys.stdout)
        print(age)
        genderstr = flask.request.form['gender']
        print('genderstr : ',file=sys.stdout)
        print(genderstr)
        if genderstr == 'M':
            gender_1 = 0
            gender_2 = 1
        else:
            gender_1 = 1
            gender_2 = 0
        print('gender : ',file=sys.stdout)
        print(gender_1,file=sys.stdout)
        print(gender_2,file=sys.stdout)
        height = float(flask.request.form['height'])
        print('height : ', file=sys.stdout)
        print(height)
        weight = float(flask.request.form['weight'])
        print('weight : ', file=sys.stdout)
        print(weight)
        ap_hi = float(flask.request.form['ap_hi'])
        print('ap_hi : ', file=sys.stdout)
        print(ap_hi)
        ap_low = float(flask.request.form['ap_low'])
        print('ap_low : ', file=sys.stdout)
        print(ap_low)
        cholesterolstr = (flask.request.form['cholesterol'])
        print('cholesterolstr : ', file=sys.stdout)
        print(cholesterolstr)
        if cholesterolstr == '1':
            cholesterol_1 = 1
            cholesterol_2 = 0
            cholesterol_3 = 0
        elif cholesterolstr == '2':
            cholesterol_1 = 0
            cholesterol_2 = 1
            cholesterol_3 = 0
        else:
            cholesterol_1 = 0
            cholesterol_2 = 0
            cholesterol_3 = 1

        print(cholesterol_1, file=sys.stdout)
        print(cholesterol_2, file=sys.stdout)
        print(cholesterol_3, file=sys.stdout)


        glucosestr = flask.request.form['glucose']
        print('glucostr : ', file=sys.stdout)
        print(glucosestr)

        if glucosestr == '1':
            gluc_1 = 1
            gluc_2 = 0
            gluc_3 = 0
        elif glucosestr == '2':
            gluc_1 = 0
            gluc_2 = 1
            gluc_3 = 0
        else:
            gluc_1 = 0
            gluc_2 = 0
            gluc_3 = 1
        print('glucose : ', file=sys.stdout)
        print(gluc_1, file=sys.stdout)
        print(gluc_2, file=sys.stdout)
        print(gluc_3, file=sys.stdout)


        smokestr = flask.request.form['smoke']
        print('smokestr : ', file=sys.stdout)
        print(smokestr)
        if smokestr == '0':
            smoke_0 = 1
            smoke_1 = 0
        else:
            smoke_0 = 0
            smoke_1 = 1
        print(smoke_0, file=sys.stdout)
        print(smoke_1, file=sys.stdout)



        alcostr = flask.request.form['alco']
        print('alcostr : ', file=sys.stdout)
        print(alcostr)
        if alcostr == '0':
            alco_0 = 1
            alco_1 = 0
        else:
            alco_0 = 0
            alco_1 = 1

        print(alco_0, file=sys.stdout)
        print(alco_1, file=sys.stdout)

        activestr = flask.request.form['active']
        print('activestr : ', file=sys.stdout)
        print(activestr)
        if activestr == '0':
            active_0 = 1
            active_1 = 0
        else:
            active_0 = 0
            active_1 = 1
        print(active_0, file=sys.stdout)
        print(active_1, file=sys.stdout)

        bmi = (703*weight)/(height*height)

        print(bmi, file=sys.stdout)
        # new_X_test_data = np.array([[age, height, weight, 110, 80, 21.948577, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]])
        new_X_test_data = np.array([[age, height, weight, ap_hi, ap_low,bmi,gender_1,gender_2,cholesterol_1,cholesterol_2, cholesterol_3, gluc_1, gluc_2, gluc_3, smoke_0,smoke_1, alco_0,alco_1, active_0, active_1]])

        SVC_pred = model_SVC.predict(new_X_test_data)
        SVC_pred_list = SVC_pred.tolist()

        KNN_pred = model_KNN.predict(new_X_test_data)
        KNN_pred_list = KNN_pred.tolist()

        LR_pred = model_LR.predict(new_X_test_data)
        LR_pred_list = LR_pred.tolist()

        RF_pred = model_RF.predict(new_X_test_data)
        RF_pred_list = RF_pred.tolist()

        XG_pred = model_XG.predict(new_X_test_data)
        XG_pred_list = XG_pred.tolist()

        NB_pred = model_NB.predict(new_X_test_data)
        NB_pred_list = NB_pred.tolist()

        # NN_pred = model_NN.predict(new_X_test_data)
        # NN_pred_list = NN_pred.tolist()

        # Accessing the first element of the list from every result list above

        svc_prediction = str(SVC_pred_list[0])
        nb_prediction = str(NB_pred_list[0])
        xg_prediction = str(XG_pred_list[0])
        knn_prediction = str(KNN_pred_list[0])
        lr_prediction = str(LR_pred_list[0])
        rf_prediction = str(RF_pred_list[0])
        # result_df = pd.DataFrame(pred)
        result = {"SVC_Prediction": svc_prediction,"NB_Prediction": nb_prediction,"XG_Prediction": xg_prediction, "KNN_Prediction": knn_prediction,"LR_Prediction": lr_prediction, "RF_Prediction": rf_prediction}
        print(result, file=sys.stdout)
        #Used json.dumps to get result rendering results.html on POST
        return render_template('results.html', data=json.dumps(result))



if __name__ == '__main__':
    app.run()