import tensorflow as tf
import tensorflow_hub as hub
from sklearn.svm import SVC
import pickle
import numpy as np

embed = hub.KerasLayer(r"C:\Users\Chinmoy Dutta\Desktop\frontend\app\model\universal-sentence-encoder-large_5")
#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
label_map = {0:'correct', 1:'incorrect', 2:'contradictory'}
with open(r'/Users/admin/Documents/frontend/app/modelsvm_model.pickle','rb') as f:
    model = pickle.load(f)

premise = input('Enter student ans:')
hypothesis = input('Reference ans:')
input_ = [premise, hypothesis]
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