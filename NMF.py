import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io.wavfile as wav


def cost_snmf(V, W, H, beta=2, mu=0.1):
    A = tf.matmul(W, H)
    tmp = W * tf.matmul(A ** (beta - 1), H.T)
    numerator = tf.matmul(A ** (beta - 2) * V, H.T) + W * (
        tf.matmul(tf.ones((tf.shape(tmp)[0], tf.shape(tmp)[0])), tmp))
    tmp2 = W * tf.matmul(A ** (beta - 2) * V, H.T)
    denumerator = tf.matmul(A ** (beta - 1), H.T) + W * (
        tf.matmul(tf.ones((tf.shape(tmp2)[0], tf.shape(tmp2)[0])), tmp2))
    W_new = numerator / denumerator

    H_new = tf.matmul(W.T, V * A ** (beta - 2)) / (tf.matmul(W.T, A ** (beta - 1)) + mu)

    return W_new, H_new


def optimize(mode, V, W, H, beta, mu, lr, const_W):
    # cost function
    cost = tf.reduce_mean(tf.square(V - tf.matmul(W, H)))

    # update operation for H
    if mode == 'snmf':
        # Sparse NMF MuR

        A = tf.matmul(W, H)
        H_new = tf.matmul(tf.transpose(W), V * A ** (beta - 2)) / (tf.matmul(tf.transpose(W), A ** (beta - 1)) + mu)
        H_update = H.assign(H_new)
    elif mode == 'nmf':
        # Basic NMF MuR
        Wt = tf.transpose(W)
        H_new = H * tf.matmul(Wt, V) / (tf.matmul(tf.matmul(Wt, W), H) + 1e-8)
        H_update = H.assign(H_new)
    elif mode == 'pg':
        """optimization; Projected Gradient method """
        dW, dH = tf.gradients(xs=[W, H], ys=cost)
        H_update_ = H.assign(H - lr * dH)
        H_update = tf.where(tf.less(H_update_, 0), tf.zeros_like(H_update_), H_update_)

    # update operation for W
    if const_W == False:
        if mode == 'snmf':
            # Sparse NMF MuR

            vec = tf.reduce_sum(W, 0)
            multiply = tf.constant([tf.shape(W)[0]])

            de = tf.reshape(tf.tile(tf.reduce_sum(W, 0), multiply), [multiply[0], tf.shape(tf.reduce_sum(W, 0))[0]])
            W = W / de
            Ht = tf.transpose(H)
            tmp = W * tf.matmul(A ** (beta - 1), Ht)
            n = tf.shape(tmp)[0]
            numerator = tf.matmul(A ** (beta - 2) * V, Ht) + W * (tf.matmul(tf.ones((n, n)), tmp))
            tmp2 = W * tf.matmul(A ** (beta - 2) * V, Ht)
            denumerator = tf.matmul(A ** (beta - 1), Ht) + W * (tf.matmul(tf.ones((n, n)), tmp2))
            W_new = W * numerator / denumerator
            W_update = W.assign(W_new)
        elif mode == 'nmf':
            # Basic NMF MuR
            Ht = tf.transpose(H)
            W_new = W * tf.matmul(V, Ht) / tf.matmul(W, tf.matmul(H, Ht))
            W_update = W.assign(W_new)
        elif mode == 'pg':
            W_update_ = W.assign(W - lr * dW)
            W_update = tf.where(tf.less(W_update_, 0), tf.zeros_like(W_update_), W_update_)

        return W_update, H_update, cost

    return 0, H_update, cost


# code from https://github.com/eesungkim/NMF-Tensorflow
def NMF_MuR(V_input, r, max_iter, display_step, const_W, init_W):
    m, n = np.shape(V_input)

    tf.reset_default_graph()

    V = tf.placeholder(tf.float32)

    initializer = tf.random_uniform_initializer(0, 1)

    if const_W == False:
        W = tf.get_variable(name="W", shape=[m, r], initializer=initializer)
        H = tf.get_variable("H", [r, n], initializer=initializer)
    else:
        W = tf.constant(init_W, shape=[m, r], name="W")
        H = tf.get_variable("H", [r, n], initializer=initializer)

    mode = 'nmf'
    W_update, H_update, cost = optimize(mode, V, W, H, beta=2, mu=0.00001, lr=0.1, const_W=const_W)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for idx in range(max_iter):
            if const_W == False:
                W = sess.run(W_update, feed_dict={V: V_input})
                H = sess.run(H_update, feed_dict={V: V_input})
            else:
                H = sess.run(H_update, feed_dict={V: V_input})

            if (idx % display_step) == 0:
                costValue = sess.run(cost, feed_dict={V: V_input})
                print("|Epoch:", "{:4d}".format(idx), " Cost=", "{:.3f}".format(costValue / 1))

    print("================= [Completed Training NMF] ===================")
    return W, H

def divide_magphase(D, power=1):
    """Separate a complex-valued stft D into its magnitude (S)
    and phase (P) components, so that `D = S * P`."""

    mag = np.abs(D)**power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase

def merge_magphase(magnitude, phase):
    return magnitude * phase