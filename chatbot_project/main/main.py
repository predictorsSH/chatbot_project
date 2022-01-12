import tensorflow as tf
from chatbot_project.preprocess.data_preprocessing import Data_preprcessing
from chatbot_project.model.transformer_model import transformer
from chatbot_project.main.custom_fnc import CustomSchedule
from chatbot_project.preprocess.preprocess_func import preprocess_sentence


BATCH_SIZE = 64
BUFFER_SIZE = 20000
FILE_PATH = '../../Dataset/ChatBotData.csv'

#데이터준비
DP = Data_preprcessing(FILE_PATH)
padded_questions, padded_answers, vocab_size, max_len, tokenizer, START_TOKEN, END_TOKEN, = DP.data_converter()


dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': padded_questions,
        'dec_inputs': padded_answers[:, :-1] # 디코더의 입력. 마지막 패딩 토큰이 제거된다.
    },
    {
        'outputs': padded_answers[:, 1:]  # 맨 처음 토큰이 제거된다. 다시 말해 시작 토큰이 제거된다.
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

#트랜스포머모델
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1
VALIDATAION_RATE =0.1
EPOCHS=50

model = transformer(
    vocab_size=vocab_size,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, max_len - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

def accuracy(y_true, y_pred):
    # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, max_len - 1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(max_len):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

check_point_file_path = '../save_model'

#custom subclass사용하면 weight only사용해야함
model_check_point = tf.keras.callbacks.ModelCheckpoint(
    filepath='../save_model',
    monitor='val_accuarcy',
    save_best_only=False,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
    options=None
)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

# history = model.fit(dataset, epochs=EPOCHS, callbacks=[model_check_point])


model.load_weights(check_point_file_path)


output = predict("멍청이!")
