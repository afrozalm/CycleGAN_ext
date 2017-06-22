import tensorflow as tf
from model import CycleEXT
from solver import Solver

flags = tf.app.flags
flags.DEFINE_integer('n_classes', 200, "number of classes")
flags.DEFINE_boolean('skip', False, "to keep skip connection in transformer")
flags.DEFINE_float('adv_weight', 1.0, "weight to adversarial loss")
flags.DEFINE_float('class_weight', 1.0, "weight to classification loss")
flags.DEFINE_float('cyc_weight', 1.0, "weight to cycle loss")
flags.DEFINE_float('ucn_weight', 1.0, "weight to ucn loss")
flags.DEFINE_float('margin', 2.0, "margin for ucn negative class")
flags.DEFINE_integer('disc_rep', 3, "disc repeats")
flags.DEFINE_integer('gen_rep', 1, "gen repeats")
flags.DEFINE_string('loss_type', 'wass', "choose from 'wass'|'cross'")

flags.DEFINE_integer('batch_size', 25, "set the value of batch_size")
flags.DEFINE_string('mode', 'train', "'pretrain', 'train' or 'eval'")
flags.DEFINE_float('learning_rate', 1e-4, "learning rate for RMSProp")
flags.DEFINE_integer('pretrain_iter', 20000, "iterations to pretrain model")
flags.DEFINE_integer('train_iter', 20000, "iterations to train model")
flags.DEFINE_integer('sample_iter', 100, "iterations to get images")
flags.DEFINE_string('pretrained_model', '',
                    "location of pretrained model")
flags.DEFINE_string('test_model', 'model/cycle-400', "location for test model")
flags.DEFINE_string('log_dir', 'logs', "location for log directory")
flags.DEFINE_string('model_save_path', 'model',
                    "directory for saving the model")
flags.DEFINE_string('sample_save_path', 'sample',
                    "directory for saving the sampled images")
FLAGS = flags.FLAGS


def main(_):

    model = CycleEXT(mode=FLAGS.mode,
                     learning_rate=FLAGS.learning_rate,
                     n_classes=FLAGS.n_classes,
                     class_weight=FLAGS.class_weight,
                     cyc_weight=FLAGS.cyc_weight,
                     ucn_weight=FLAGS.ucn_weight,
                     skip=FLAGS.skip,
                     margin=FLAGS.margin,
                     loss_type=FLAGS.loss_type)

    solver = Solver(model, batch_size=FLAGS.batch_size,
                    pretrain_iter=FLAGS.pretrain_iter,
                    train_iter=FLAGS.train_iter, sample_iter=100,
                    real_dir='real-face', caric_dir='caricature-face',
                    combined_dir='class-combined.pkl',
                    log_dir=FLAGS.log_dir,
                    model_save_path=FLAGS.model_save_path,
                    sample_save_path=FLAGS.sample_save_path,
                    pretrained_model=FLAGS.pretrained_model,
                    test_model=FLAGS.test_model,
                    disc_rep=FLAGS.disc_rep,
                    gen_rep=FLAGS.gen_rep)

    # create directories if not exist
    if not tf.gfile.Exists(FLAGS.model_save_path):
        tf.gfile.MakeDirs(FLAGS.model_save_path)
    if not tf.gfile.Exists(FLAGS.sample_save_path):
        tf.gfile.MakeDirs(FLAGS.sample_save_path)

    if FLAGS.mode == 'pretrain':
        solver.pretrain()
    elif FLAGS.mode == 'train':
        solver.train()
    else:
        solver.eval()


if __name__ == '__main__':
    tf.app.run()
