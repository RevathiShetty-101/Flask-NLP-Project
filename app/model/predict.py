from tensorflow.keras.models import load_model
import Prepare
import WordEmbedding
import SiameseModel


word_embedding_file = 'glove_vector_dict_300d.pickle'
model_file = 'SRA_trained_model.h5'

word_embedding = WordEmbedding(word_embedding_file)
input_ = Prepare(data_frame)
label_map = {0:'correct', 1:'incorrect', 2:'contradictory'}

model = SiameseModel(word_embedding, input_)
model.load_weights(model_file, by_name=False, skip_mismatch=False)
predictions = model.predict(input_.premise, input_.hypothesis)

predicted_labels = [label_map[np.argmax(p)] for p in predictions]