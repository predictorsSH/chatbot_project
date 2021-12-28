import tensorflow as tf

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :]

if __name__ == '__main__' :
    print(create_padding_mask(tf.constant([[1, 21, 777, 0, 0]])))
    