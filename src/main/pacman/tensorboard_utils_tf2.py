import tensorflow as tf


def create_summary_writer(logdir):
    return tf.summary.create_file_writer(logdir)


def create_summary(tag, value, step):
    return tf.summary.scalar(tag, value, step=step)


def add_summary(writer, tag, value, step):
    with writer.as_default():
        create_summary(tag, value, step)
        writer.flush()


def add_summaries(writer, tags, values, step, prefix=''):
    for (t, v) in zip(tags, values):
        with writer.as_default():
            create_summary(prefix + t, v, step)
            writer.flush()