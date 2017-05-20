import tensorflow as tf

with tf.Graph().as_default():
    with tf.Session() as sess:
        x = tf.constant([[1, 2, 3, 9], [4, 5, 6, 9]])

        # y = tf.constant([[88], [88]])
        y = tf.fill([2, 1], 88, name=None)

        # z = sess.run(tf.concat([x, y], 1))
        # print(z)

        # r =  tf.placeholder(tf.int32, shape=[2, None], name="r")
        # for i in range(x.shape[0]):
        #     x[i] = tf.concat([x[i], [88]], 0)

        r = tf.concat([y, x], 1)
        z = sess.run(r)
        print(z)

        print("\n\n slice:")
        r = tf.slice(r, [0, 0], [-1, int(r.shape[1] - 1)])
        print(sess.run(r))
