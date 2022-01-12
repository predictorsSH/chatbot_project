from chatbot_project.preprocess.preprocess_func import *
import tensorflow_datasets as tfds
import tensorflow as tf

class Data_preprcessing():

    def __init__(self, file_path,  max_len:int = 40):
        self.file_path = file_path
        self.max_len = max_len

    def data_converter(self,):

        data = data_load(self.file_path)

        questions, answers = make_dataset(data)

        tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = self.tokenizing(questions, answers)

        encoded_inputs, encoded_outputs = self.encoding(questions,answers,tokenizer, START_TOKEN, END_TOKEN)

        padded_inputs, padded_outputs = self.padding(encoded_inputs,encoded_outputs)

        MEX_LEN = self.max_len

        return padded_inputs, padded_outputs, VOCAB_SIZE, MEX_LEN, tokenizer, START_TOKEN, END_TOKEN,

    def tokenizing(self,questions, answers):
        tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = tokenizer_(questions, answers)


        return tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE

    def encoding(self, questions, answers, tokenizer ,START_TOKEN, END_TOKEN):
        encoded_inputs=[]
        encoded_outputs=[]

        for q,a in zip(questions, answers):
            q=START_TOKEN + tokenizer.encode(q) + END_TOKEN
            a=START_TOKEN + tokenizer.encode(a) + END_TOKEN

            encoded_inputs.append(q)
            encoded_outputs.append(a)

        return encoded_inputs, encoded_outputs

    def padding(self, encoded_inputs, encoded_outputs):

        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(encoded_inputs, maxlen=self.max_len,padding='post')
        padded_outputs = tf.keras.preprocessing.sequence.pad_sequences(encoded_outputs,maxlen=self.max_len,padding='post')

        return padded_inputs, padded_outputs

FILE_PATH = '../../Dataset/ChatBotData.csv'

if __name__ == '__main__':
    DP = Data_preprcessing(FILE_PATH)
    padded_inputs, padded_outputs, vocab_size  = DP.data_converter()


