# coding: utf8
import tensorflow as tf


class Model:

    def __init__(self, model_path=None):
        # computation graph
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # graph initialization
        # case where we update a previous graph
        if model_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            print("\nLoading weights from a previous trained model at " + model_path + " !!!")
            self.saver.restore(self.sess, model_path)

    def fit(self):

        print('fit')

    def validate(self):

        print('validate')

    def test(self):

        print('test')

    def close(self, path):

        self.saver.save(self.sess, path)
        print("\nmodel is saved at ", path, " !!!")
        self.sess.close()
        tf.reset_default_graph()

    def reset(self):

        self.sess.close()
        tf.reset_default_graph()
