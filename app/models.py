from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from app.db_create import Base

metadata = Base.metadata


class BigQuestion(Base):
    __tablename__ = 'BigQuestion'

    q_id = Column(Integer, primary_key=True)
    q_desc = Column(String(100))
    ref_ans = Column(String(1000))

    def __init__(self,q_id,q_desc,ref_ans):
        self.q_id = q_id
        self.q_desc = q_desc
        self.ref_ans = ref_ans

class Answer(Base):
    __tablename__ = 'Answer'
    
    Ans_id = Column(Integer, primary_key=True, nullable=False)
    b1 = Column(String(1000))

    def __init__(self,Ans_id,b1):
        self.Ans_id=Ans_id
        self.b1 = b1

class res(Base):
    __tablename__ = 'res'
    
    user_id=Column(Integer,primary_key=True)
    feedback=Column(String(1000))   
    
    def __init__(self,user_id,feedback):
        self.user_id = user_id
        self.feedback = feedback
    
    


class Test(Base):
    __tablename__ = 'Test'

    test_id = Column(Integer, primary_key=True)
    test_name = Column(String(30), nullable=False)
    big1 = Column(Integer)

    def __init__(self,test_id,test_name,big1):
        self.test_id = test_id
        self.test_name = test_name
        self.big1 = big1


class User(Base):
    __tablename__ = 'User'

    name = Column(String(30), nullable=False)
    email_id = Column(String(30), nullable=False, unique=True)
    user_id = Column(Integer, primary_key=True)
    password = Column(String(30), nullable=False)
    user_type = Column(String(30), nullable=False)

    def __init__(self,name,email_id,user_id,password,user_type):
        self.name = name
        self.email_id = email_id
        self.user_id = user_id
        self.password = password
        self.user_type = user_type

    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        try:
            return self.user_id  
        except NameError:
            return self.user_id 
