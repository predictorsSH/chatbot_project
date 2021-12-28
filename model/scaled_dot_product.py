import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)

    # Q와 K의 곱. 어텐션 스코어 행렬.
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 스케일링
    # dk의 루트값으로 나눠준다.
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        logits += (mask * -1e9)

    # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
    # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights



import numpy as np
if __name__ == '__main__':
    # 임의의 Query, Key, Value인 Q, K, V 행렬 생성
    np.set_printoptions(suppress=True)
    temp_k = tf.constant([[10,0,0],
                          [0,10,0],
                          [0,0,10],
                          [0,0,10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[   1,0],
                          [  10,0],
                          [ 100,5],
                          [1000,6]], dtype=tf.float32)  # (4, 2)
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
    print(temp_out) # 어텐션 값

    # Query의 값만 다른 값으로 바꿔보고 함수를 실행

    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
    print(temp_out) # 어텐션 값


    # 3개의 Query의 값을 함수의 입력으로 사용하여 확인

    temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
    print(temp_out) # 어텐션 값
