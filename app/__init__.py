import os
from flask_login import LoginManager
from flask_openid import OpenID
from flask import Flask, render_template, json, request, redirect, session
from flaskext.mysql import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from app.model import Prepare,WordEmbedding


app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
lm = LoginManager()
lm.init_app(app)


from app import views, models