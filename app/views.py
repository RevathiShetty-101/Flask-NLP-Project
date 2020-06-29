import pandas as pd
from tensorflow.keras.models import load_model
from app import app,models,lm
import os
from flask_login import LoginManager,login_required,login_user,logout_user
from app.db_create import db_session
from flask_openid import OpenID
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, flash, render_template, json, request, redirect, session, url_for
from flaskext.mysql import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from app.models import User,Test,Answer,BigQuestion,res
from app.model.Prepare import Prepare
from app.model.WordEmbedding import WordEmbedding
from app.model.SiameseModel import SiameseModel
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.svm import SVC

@app.route('/')
def main():
    return render_template('index.html')

# Staff Views

@app.route('/showStaffDash')
@login_required
def showStaffDash():
	lst = Test.query.all()
	return render_template('staffdash.html',lst=lst)

@app.route('/showAddTest',methods=['GET','POST'])
@login_required
def showAddTest():
    bq = BigQuestion.query.all()
    qp = request.form.get('question')
    return render_template('addtest.html',bq=bq)

@app.route('/showBigQ')
@login_required
def showBigQ():
    return render_template('bigquestion.html')
    
@app.route('/showResult',methods=['POST'])
@login_required
def showResult():
    u_id = request.form['userid']
    lst = res.query.filter(res.user_id == u_id)
    return render_template('results.html',lst=lst)
    

@app.route('/addBigQ',methods = ['POST'])
@login_required
def addBigQ():
    q_id = request.form['questid']
    q_desc = request.form['questdesc']
    ref_ans = request.form['ref_ans']
    q = BigQuestion(q_id,q_desc,ref_ans)
    db_session.add(q)
    db_session.commit()
    flash("Successfully Added.")
    return redirect(url_for('showBigQ'))

@app.route('/addTest',methods = ['POST'])
@login_required
def addTest():
    test_id = request.form['testid'] 
    test_name = request.form['testname']
    Questions = request.form['bq1']
    t = Test(test_id,test_name,Questions)
    db_session.add(t)
    db_session.commit()
    flash("Successfully Added.")
    return redirect(url_for('showAddTest'))

@app.route('/showStudentDash')
@login_required
def showStudentDash():
    lst= Test.query.all()
    #bst= BigQuestion.query.all()
    ans = Answer.query.all()
    return render_template('studentdash.html',lst=lst,ans=ans)

@app.route('/showAttendTest', methods = ['POST','GET'])
@login_required
def showAttendTest():
    test_id = request.form['test_id']
    t = Test.query.get(test_id)
    b1 = BigQuestion.query.get(t.Big1)
    return render_template('attendtest.html',t=t,b1=b1)

@app.route('/validate', methods=['POST'])
@login_required
def validate():
    if request.method == "POST":
       Ans_id = request.form['userid']
       b11 = request.form['b1']
       #b22 = request.form['b2']
       r = Answer(Ans_id,b11)
       db_session.add(r)
       db_session.commit()
       q_id=request.form['tesid']
       #a_id=request.form['teid']
       embed = hub.KerasLayer(r"/Users/admin/Documents/frontend/app/model/universal-sentence-encoder-large_5")
       #embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
       label_map = {0:'correct', 1:'incorrect', 2:'contradictory'}
       with open(r'/Users/admin/Documents/frontend/app/model/model.pickle','rb') as f:
            model = pickle.load(f)
            u=Answer.query.filter(Answer.Ans_id==Ans_id).first()
            df=np.array(u.b1)
            #df1=np.array(u.b2)
            #print(df)
            #print(df1)
            b=BigQuestion.query.filter(BigQuestion.q_id==q_id).first()
            rf=np.array(b.ref_ans)
            #print(rf)
            #b1=BigQuestion.query.filter(BigQuestion.q_id==a_id).first()
            #rf1=np.array(b1.ref_ans)
            #print(rf1)
            '''
            data={'premise':[df,df1],'hypothesis':[rf]}
            data_frame=pd.DataFrame(data,columns=['premise','hypothesis'])
            premise=data_frame.premise.tolist()
            hypothesis=data_frame.hypothesis.tolist()
            model_input=np.empty((0,1024))
            '''
            premise=np.array(df)
            print("Student Answer:",premise)
            hypothesis=np.array(rf)
            print("Reference Answer:",hypothesis)
            input_=[premise, hypothesis]
            #print(input_)
            '''
            for i in range(len(premise)):
                r=np.empty((0,512))
                input_=[premise[i],hypothesis[i]]
            '''
            embeddings = []
            r = np.empty((0,512))
            for test in input_:
                r = np.vstack((r, np.array(embed([test]))))
            embeddings = r
            print(embeddings.shape)
            x1 = np.multiply(embeddings[0], embeddings[1])
            x2 = np.absolute(embeddings[0]-embeddings[1])

            x = np.hstack((x1,x2))
            model_input = np.empty((0,1024))
            model_input = np.vstack((model_input,x))
            print(model_input.shape)

            prediction = model.predict(model_input)
            print(prediction)
            inverse_label={0:'correct',1:'incorrect',2:'contradictory'}
            #print(prediction)
            feedback=inverse_label[prediction[0]]
            print(feedback)
            user_id = request.form['userid']
            a=res(user_id,feedback)
            db_session.add(a)
            db_session.commit()
    return render_template('Result.html',feedback=feedback)        
                    
        


@app.route('/showStaffProfile')
def showStaffProfile():
	return render_template('staffprofile.html')


@app.route('/showStudentProfile')
def showStudentProfile():
	return render_template('studentprofile.html')

#Login or Signup views

@app.route('/showSignUp')
def showSignUp():
    return render_template('signup.html')

@app.route('/signUp',methods = ['POST','GET'])
def signUp():
    if request.method == 'POST':
        name = request.form ['usernamesignup']
        email_id = request.form ['emailsignup']
        user_id = request.form ['useridsignup']
        password = request.form ['passwordsignup']
        user_type = request.form ['user_typesignup']
        if name and email_id and user_id and password and user_type:
            u = User(name,email_id,int(user_id),password,user_type)
            db_session.add(u)
            db_session.commit() 
            flash("Successfully Registered.")
            return redirect(url_for('showSignUp'))
        else:
            flash("Please enter all the details.")
            return redirect(url_for('showSignUp'))
    else:
        return redirect(url_for('showSignUp'))

@app.route('/showSignin')
def showSignin():
	return render_template('signup.html')

@app.route('/signIn',methods = ['POST','GET'])
def signIn():
    if request.method == 'POST':
        user_id = request.form ['useridsignin']
        password = request.form ['passwordsignin']
        u = User.query.filter(User.user_id == user_id)
        user=None
    for i in u:
        if i.password == password:
            user = i
            if not user:
                flash("Incorrect User ID or Password.") 
                return redirect(url_for('showSignin'))
            else:
                login_user(user)
                if user.user_type == "student":
                    return redirect(url_for('showStudentDash'))
                else:
                    return redirect(url_for('showStaffDash'))
        else:
            return redirect(url_for('showSignin'))
		

@lm.user_loader
def load_user(user_id):
	user = None
	u = User.query.filter(User.user_id == user_id)
	for i in u:
		if i.user_id == user_id:
			user = i
			break
	return user

@app.route('/logout')
@login_required
def logout():
	logout_user()
	return redirect(url_for('main'))
    
