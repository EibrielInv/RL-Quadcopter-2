class Logger:
    def info(self, message):
        print(message)

    def get_dir(self):
        return None


logger = Logger()


class dummy_U:
    def single_threaded_session(self):
        num_cpu = 1
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        return tf.Session(config=tf_config)


U = dummy_U()
