import os
basedir = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = 'mysql://root:root@localhost/stud'
SQLALCHEMY_TRACK_MODIFICATIONS = True

WTF_CSRF_ENABLED = True
SECRET_KEY = 'something very secretive and no one knows it!!'

#MYSQL_DATABASE_USER = 'root'
#MYSQL_DATABASE_PASSWORD = 'root'
#MYSQL_DATABASE_DB = 'exam'
#MYSQL_DATABASE_HOST = 'localhost'