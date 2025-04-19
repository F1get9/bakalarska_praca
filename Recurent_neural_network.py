import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import regex as re
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def file_to_sentence_list(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sentences

file_path = 'kingdom.txt'
text_data = file_to_sentence_list(file_path)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

input_layer = Input(shape=(max_sequence_len - 1,))
embedding_layer = Embedding(total_words, 10, input_length=max_sequence_len - 1, name='embedding')(input_layer)
lstm_layer = LSTM(128, name='lstm')(embedding_layer)
output_layer = Dense(total_words, activation='softmax', name='dense')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_model = Model(inputs=model.input, outputs=model.get_layer('lstm').output)
dense_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)

watched_words = ['king','queen','man','woman']
watched_indices = [tokenizer.word_index[word] for word in watched_words]
padded = pad_sequences([[idx] for idx in watched_indices], maxlen=max_sequence_len-1, padding='pre')

paraphrases = [
    "king and queen shared harmony and unity.",
    "harmony and unity is shared by king and queen.",
    "man and woman stand equal.",
    "equal stand between man and woman."
]

history_input_embed = []
history_lstm_embed = []
history_output_embed = []

for epoch in range(200):
    model.fit(X, y, epochs=1, verbose=1)
    paraphrase_seqs = tokenizer.texts_to_sequences(paraphrases)
    paraphrase_padded = pad_sequences(paraphrase_seqs, maxlen=max_sequence_len - 1, padding='pre')

    embedding_weights = model.get_layer('embedding').get_weights()[0]
    input_embeds = np.array([embedding_weights[idx] for idx in watched_indices])
    mean, eig = cv2.PCACompute(input_embeds, None, maxComponents=2)
    reduced_input = np.dot(input_embeds - mean, eig.T)
    history_input_embed.append((reduced_input,epoch))
    
    lstm_out = lstm_model.predict(paraphrase_padded)
    mean, eig = cv2.PCACompute(lstm_out, None, maxComponents=2)
    reduced_lstm = np.dot(lstm_out - mean, eig.T)
    history_lstm_embed.append((reduced_lstm,epoch))

    dense_out = dense_model.predict(padded)
    mean, eig = cv2.PCACompute(dense_out, None, maxComponents=2)
    reduced_dense = np.dot(dense_out - mean, eig.T)
    history_output_embed.append((reduced_dense,epoch))

model.save('kingdom.h5')

def make_video_lstm(history, filename, title):
    out = cv2.VideoWriter()
    out.open(filename, cv2.VideoWriter_fourcc(*'MJPG'), 10, (1600, 800))
    colors = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255)]
    for codes in history:
        disp = np.zeros((800, 1600, 3), np.uint8)
        _ = cv2.putText(disp, paraphrases[0][:] , (180,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[0], 1)
        _ = cv2.putText(disp, paraphrases[1][:] , (180,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[1], 1)
        _ = cv2.putText(disp, paraphrases[2][:] , (180,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[2], 1)
        _ = cv2.putText(disp, paraphrases[3][:] , (180,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[3], 1)
        _ = cv2.circle(disp, (800, 400), 300, (255, 255, 255), 1)
        _ = cv2.putText(disp, title, (1200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        _ = cv2.putText(disp, f'epoch: {codes[1]}', (1200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        for i, (code, phrase, color) in enumerate(zip(codes[0], paraphrases, colors)):
            x, y = code
            xx, yy = int(x * 60 + 800), 799 - int(y * 60 + 400)
            _ = cv2.circle(disp, (xx, yy), 6, color, cv2.FILLED)
        out.write(disp)
    out.release()

make_video_lstm(history_lstm_embed, 'paraphrase_lstm.avi', 'LSTM Output')

def make_video_words(history, filename):
    out = cv2.VideoWriter()
    out.open(filename, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1600,800))
    colors = [(0,0,255),(0,255,0),(255,255,0),(0,255,255)]

    for codes in history:
        disp = np.zeros((800,1600,3), np.uint8)
        _ = cv2.circle(disp,(800,400),300,(255,255,255),1)
        _ = cv2.putText(disp, f'epoch: {codes[1]}', (1200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
        for code, word, color in zip(codes[0], watched_words, colors):
            x, y = code
            xx, yy = int(x*200+800), 799-int(y*200+400)
            _ = cv2.circle(disp, (xx,yy), 5, color, cv2.FILLED)
            _ = cv2.putText(disp, word, (xx+5,yy-5), 0, 1.0, color, 2)
        out.write(disp)
    out.release()

make_video_words(history_input_embed, 'embedding_input.avi')
make_video_words(history_output_embed, 'embedding_output.avi')
