
from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import tensorflow
from flask import send_from_directory

app = Flask(__name__, template_folder = "template")


@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")
 
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")


"""
@app.route("/register", methods = ["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # flash("Account created for {form.username.data}!".format("success"))
        flash("Account created", "success")      
        return redirect(url_for("home"))
    return render_template("register.html", title = "Register", form = form )
@app.route("/login", methods = ["POST", "GET"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # if form.email.data == "sho" and form.password.data == "password":
        flash("You Have Logged in !", "success")
        return redirect(url_for("home"))
    # else:
    # flash("Login Unsuccessful. Please check username and password", "danger")
    return render_template("login.html", title = "Login", form = form )
def ValuePredictor1(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 30)
    loaded_model = joblib.load("model")
    result = loaded_model.predict(to_predict)
    return result[0]
    
@app.route("/result1", methods = ["GET", "POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = "cancer"
        else:
            prediction = "Healthy"       
    return(render_template("result.html", prediction = prediction))"""


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if(size == 8): # Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif(size == 30): # Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size == 12): # Kidney
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif(size == 10): # Liver
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif(size == 11): # Heart
        loaded_model = joblib.load("model2")
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/result", methods = ["POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list) == 30): # Cancer
            result = ValuePredictor(to_predict_list, 30)
        elif(len(to_predict_list) == 8): # Daiabtes
            result = ValuePredictor(to_predict_list, 8)
        elif(len(to_predict_list) == 12):
            result = ValuePredictor(to_predict_list, 12)
        elif(len(to_predict_list) == 11):
            result = ValuePredictor(to_predict_list, 11)
            # if int(result) == 1:
            #   prediction = "diabetes"
            # else:
            #   prediction = "Healthy" 
        elif(len(to_predict_list) == 10):
            result = ValuePredictor(to_predict_list, 10)
    if(int(result) == 1):
        prediction = "Sorry ! Suffering"
    else:
        prediction = "Congrats ! you are Healthy" 
    return(render_template("result.html", prediction = prediction))


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = "8080")