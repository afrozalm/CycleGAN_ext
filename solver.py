import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle
import os
import scipy.io
import scipy.misc
from data_loader import DataLoader


class Solver(object):

    def __init__(self, model, batch_size=100, pretrain_iter=20000,
                 train_iter=2000, sample_iter=100,
                 real_dir='real-face', caric_dir='caricature-face',
                 combined_dir='class-combined.pkl',
                 log_dir='logs', n_classes=200,
                 sample_save_path='sample',
                 model_save_path='model',
                 pretrained_model='model/pre_model-4000',
                 test_model='model/hirar-400',
                 disc_rep=1,
                 gen_rep=1):

        self.loader = DataLoader(batch_size)
        self.model = model
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.pretrain_iter = pretrain_iter
        self.combined_dir = combined_dir
        self.train_iter = train_iter
        self.sample_iter = sample_iter
        self.real_dir = real_dir
        self.caric_dir = caric_dir
        self.log_dir = log_dir
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.disc_rep = disc_rep
        self.gen_rep = gen_rep

    def load_real(self, image_dir, split='train'):
        print ('loading real faces..')
        image_file = 'train.pkl' if split == 'train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            real_faces = pickle.load(f)
        mean = np.mean(real_faces['X'])
        images = real_faces['X'] / mean - 1
        labels = real_faces['y']
        print ('finished loading real faces..!')
        return images, labels

    def load_caric(self, image_dir, split='train'):
        print ('loading caricature faces..')
        image_file = 'train.pkl' if split == 'train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            caric = pickle.load(f)
        mean = np.mean(caric['X'])
        images = caric['X'] / mean - 1
        labels = caric['y']
        print ('finished loading caricature faces..!')
        return images, labels

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row * h, row * w * 2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h, :] = s
            merged[i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h, :] = t
        return merged

    def pretrain(self):

        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        # load real faces
        train_images_r, train_labels_r = self.load_real(self.real_dir,
                                                        split='train')
        test_images_r, test_labels_r = self.load_real(self.real_dir,
                                                      split='test')
        train_images_c, train_labels_c = self.load_caric(self.caric_dir,
                                                         split='train')
        test_images_c, test_labels_c = self.load_caric(self.caric_dir,
                                                       split='test')

        self.loader.add_dataset('caric_faces_tr', train_images_c)
        self.loader.add_dataset('caric_labels_tr', train_labels_c)
        self.loader.add_dataset('real_faces_tr', train_images_r)
        self.loader.add_dataset('real_labels_tr', train_labels_r)
        self.loader.link_datasets('caric_tr',
                                  ['caric_labels_tr', 'caric_faces_tr'])
        self.loader.link_datasets('real_tr',
                                  ['real_labels_tr', 'real_faces_tr'])
        self.loader.add_dataset('caric_faces_te', test_images_c)
        self.loader.add_dataset('caric_labels_te', test_labels_c)
        self.loader.add_dataset('real_faces_te', test_images_r)
        self.loader.add_dataset('real_labels_te', test_labels_r)
        self.loader.link_datasets('caric_te',
                                  ['caric_labels_te', 'caric_faces_te'])
        self.loader.link_datasets('real_te',
                                  ['real_labels_te', 'real_faces_te'])

        # build a graph
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            if self.pretrained_model != '':
                print ('loading pretrained model ..')
                saver.restore(sess, self.pretrained_model)
            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())

            for step in range(self.pretrain_iter + 1):
                caric_labels, caric_images = \
                    self.loader.next_group_batch('caric_tr')
                real_labels, real_images = \
                    self.loader.next_group_batch('real_tr')
                feed_dict = {model.real_images: real_images,
                             model.real_labels: real_labels,
                             model.caric_images: caric_images,
                             model.caric_labels: caric_labels}
                sess.run(model.train_op, feed_dict)

                if (step + 1) % 10 == 0:
                    summary, l, acc = sess.run([model.summary_op,
                                                model.loss, model.accuracy],
                                               feed_dict)

                    caric_labels, caric_images = \
                        self.loader.next_group_batch('caric_tr')
                    real_labels, real_images = \
                        self.loader.next_group_batch('real_tr')
                    test_acc, _ = \
                        sess.run(fetches=[model.accuracy, model.loss],
                                 feed_dict={model.real_images: real_images,
                                            model.real_labels: real_labels,
                                            model.caric_images: caric_images,
                                            model.caric_labels: caric_labels})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]'
                           % (step + 1, self.pretrain_iter, l, acc, test_acc))

                if (step + 1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path,
                                                  'pre_model'),
                               global_step=step + 1)
                    print ('pre_model-%d saved..!' % (step + 1))

    def train(self):
        # make directory if not exists
        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)

        # load faces
        real_images, real_labels = self.load_real(self.real_dir,
                                                  split='train')
        caric_images, caric_labels = self.load_caric(self.caric_dir,
                                                     split='train')

        self.loader.add_dataset('real_images', real_images)
        self.loader.add_dataset('caric_images', caric_images)
        self.loader.add_dataset('real_labels', real_labels)
        self.loader.add_dataset('caric_labels', caric_labels)
        self.loader.link_datasets('real', ['real_labels', 'real_images'])
        self.loader.link_datasets('caric', ['caric_labels', 'caric_images'])

        # build a graph
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            # initialize
            tf.global_variables_initializer().run()

            # restore variables of F and G
            pretrained_scopes = ['encoder_caric', 'encoder_real',
                                 'decoder_caric']
            print ('loading pretrained model ..')
            for scope in pretrained_scopes:
                variables_to_restore = \
                    slim.get_model_variables(scope=scope)
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, self.pretrained_model)

            summary_writer = tf.summary.FileWriter(
                logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('start training..!')

            for step in range(self.train_iter + 1):

                real_labels, real_images = \
                    self.loader.next_group_batch('real')
                caric_labels, caric_images = \
                    self.loader.next_group_batch('caric')

                feed_dict = {model.real_images: real_images,
                             model.real_labels: real_labels,
                             model.caric_labels: caric_labels,
                             model.caric_images: caric_images}

                for _ in xrange(self.disc_rep):
                    sess.run(model.disc_op, feed_dict)
                for _ in xrange(self.gen_rep):
                    sess.run([model.trans_op, model.dec_op], feed_dict)

                if (step + 1) % 10 == 0:
                    summary, discl, trl, decl, gl = \
                        sess.run([model.summary_op,
                                  model.loss_disc,
                                  model.loss_transformer,
                                  model.loss_decoder,
                                  model.loss_gen],
                                 feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('[Source] step: [%d/%d] disc_loss: [%.6f] \
trans_loss: [%.6f] dec_loss: [%.6f] gen_loss: [%.6f]'
                           % (step + 1, self.train_iter, discl, trl, decl, gl))

                if (step + 1) % 200 == 0:
                    saver.save(sess, os.path.join(
                        self.model_save_path, 'hirar'), global_step=step + 1)
                    print ('model/hirar-%d saved' % (step + 1))

                if (step + 1) % 5000 == 0:
                    for i in range(self.sample_iter):
                        # train model for source domain S
                        batch_images = self.loader.next_batch('real_images')
                        feed_dict = {model.real_images: batch_images}
                        sampled_batch_images = sess.run(model.trans_reconst,
                                                        feed_dict)

                        # merge and save source images and sampled target image
                        merged = self.merge_images(batch_images,
                                                   sampled_batch_images)
                        path = os.path.join(self.sample_save_path,
                                            'sample-%d-to-%d.png' %
                                            (i * self.batch_size,
                                             (i + 1) * self.batch_size))
                        scipy.misc.imsave(path, merged)
                        print ('saved %s' % path)

    def eval(self):
        # build model
        model = self.model
        model.build_model()

        # load real faces
        real_images, _ = self.load_real(self.real_dir)
        self.loader.add_dataset(name='real_images',
                                data_ptr=real_images)

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(self.sample_iter):
                # train model for source domain S
                batch_images = self.loader.next_batch('real_images')
                feed_dict = {model.images: batch_images}
                sampled_batch_images = sess.run(model.sampled_images,
                                                feed_dict)

                # merge and save source images and sampled target images
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path,
                                    'sample-%d-to-%d.png' %
                                    (i * self.batch_size, (i + 1) * self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' % path)
