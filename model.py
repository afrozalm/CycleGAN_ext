import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer


class CycleEXT(object):
    '''
    CycleGAN extension with UCN
    '''

    def __init__(self, mode='train', learning_rate=0.0003,
                 n_classes=10, class_weight=1.0, feat_layer=5,
                 skip=True, skip_layers=2, margin=4.0,
                 ucn_weight=1.0, loss_type='wass'):

        assert loss_type in ['wass', 'cross']
        self.mode = mode
        self.margin = margin
        self.ucn_weight = ucn_weight
        self.loss_type = loss_type
        self.skip = skip
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.class_weight = class_weight
        self.feat_layer = feat_layer
        self.depth_dict = {1: 64,
                           2: 128,
                           3: 256,
                           4: 512,
                           5: 512}

    def classifier(self, encodings, reuse=False):
        with tf.variable_scope('classifier', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=2,
                                weights_initializer=xavier_initializer(),
                                is_training=self.mode in ['train',
                                                          'pretrain']):

                flattened = slim.flatten(encodings)
                l1_ = slim.fully_connected(flattened, 400, scope='fc1')
                l1 = slim.dropout(l1_, scope='dropout1')
                l2_ = slim.fully_connected(l1, self.n_classes, scope='fc2')
                l2 = slim.dropout(l2_, scope='dropout2')
                return l2

    def generator(self, images, reuse=False, scope='Real2Caric'):

        assert scope in ['Real2Caric', 'Caric2Real']
        # images: (batch, 64, 64, 3) or (batch, 64, 64, 1)
        if images.get_shape()[3] == 1:
            # Replicate the gray scale image 3 times.
            images = tf.image.grayscale_to_rgb(images)

        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=2,
                                weights_initializer=xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95,
                                    center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=self.mode in ['train',
                                                              'pretrain']):
                    # (batch_size, 32, 32, 64)
                    e1_ = slim.conv2d(images, 64, [3, 3],
                                      scope='conv1')
                    e1 = slim.batch_norm(e1_, scope='g_bn1')
                    # (batch_size, 16, 16, 128)
                    e2_ = slim.conv2d(e1, 128, [3, 3],
                                      scope='conv2')
                    e2 = slim.batch_norm(e2_, scope='g_bn2')
                    # (batch_size, 8, 8, 256)
                    e3_ = slim.conv2d(e2, 256, [3, 3],
                                      scope='conv3')
                    e3 = slim.batch_norm(e3_, scope='g_bn3')
                    # (batch_size, 4, 4, 512)
                    e4_ = slim.conv2d(e3, 512, [3, 3],
                                      scope='conv4')
                    e4 = slim.batch_norm(e4_, scope='g_bn4')
                    # (batch_size, 1, 1, 512)
                    e5 = slim.conv2d(e4, 512, [4, 4], padding='VALID',
                                     scope='conv5', activation_fn=tf.nn.relu)

            with slim.arg_scope([slim.conv2d_transpose],
                                padding='SAME', activation_fn=None,
                                stride=2,
                                weights_initializer=xavier_initializer()):
                with slim.arg_scope([slim.batch_norm],
                                    decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=(self.mode == 'train')):

                    # (batch, 1, 1, 512) -> (batch_size, 4, 4, 512)
                    d1_ = slim.conv2d_transpose(e5, 512, [4, 4],
                                                padding='VALID',
                                                scope='conv_transpose1')
                    d1_ = slim.batch_norm(d1_, scope='d_bn1')
                    d1 = slim.dropout(d1_, scope='dropout1')
                    if self.skip:
                        d1 += e4

                    # (batch_size, 4, 4, 512) -> (batch_size, 8, 8, 256)
                    d2_ = slim.conv2d_transpose(d1, 256, [3, 3],
                                                scope='conv_transpose2')
                    d2_ = slim.batch_norm(d2_, scope='d_bn2')
                    d2 = slim.dropout(d2_, scope='dropout2')
                    if self.skip:
                        d2 += e3

                    # (batch_size, 8, 8, 256) -> (batch_size, 16, 16, 128)
                    d3_ = slim.conv2d_transpose(d2, 128, [3, 3],
                                                scope='conv_transpose3')
                    d3_ = slim.batch_norm(d3_, scope='d_bn3')
                    d3 = slim.dropout(d3_, scope='dropout3')
                    if self.skip:
                        d3 += e2

                    # (batch_size, 16, 16, 128) -> (batch_size, 32, 32, 64)
                    d4_ = slim.conv2d_transpose(d3, 64, [3, 3],
                                                scope='conv_transpose4')
                    d4 = slim.batch_norm(d4_, scope='d_bn4')
                    if self.skip:
                        d4 += e1

                    # (batch_size, 32, 32, 64) -> (batch_size, 64, 64, 3)
                    d5 = slim.conv2d_transpose(d4, 3, [3, 3],
                                               activation_fn=tf.nn.tanh,
                                               scope='conv_transpose5')
                    return e5, d5

    def discriminator(self, images, scope='Real', reuse=False):

        # images: (batch, 64, 64, 3)
        assert scope in ['Real', 'Caric']
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                activation_fn=None,
                                stride=2,
                                weights_initializer=xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95,
                                    center=True, scale=True,
                                    activation_fn=tf.nn.relu,
                                    is_training=(self.mode == 'train')):

                    # (batch, 64, 64, 3) -> (batch_size, 32, 32, 64)
                    net = slim.conv2d(images, 64, [3, 3],
                                      scope='conv1')
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.dropout(net, scope='dropout1')
                    # (batch_size, 32, 32, 64) -> (batch_size, 16, 16, 128)
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.dropout(net, scope='dropout2')
                    # (batch_size, 16, 16, 128) -> (batch_size, 8, 8, 256)
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3')
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.dropout(net, scope='dropout3')
                    # (batch_size, 8, 8, 256) -> (batch_size, 4, 4, 512)
                    net = slim.conv2d(net, 512, [3, 3], scope='conv4')
                    net = slim.batch_norm(net, scope='bn4')
                    net = slim.dropout(net, scope='dropout4')
                    if self.loss_type == 'cross':
                        return tf.nn.sigmoid(net)
                    return net

    def gan_disc_loss(self, real_score, fake_score):
        if self.loss_type == 'wass':
            return -tf.reduce_mean(real_score) + tf.reduce_mean(fake_score)
        else:
            EPS = 1e-12
            return tf.reduce_mean(-(tf.log(real_score + EPS)
                                    + tf.log(1 - fake_score + EPS)))

    def gan_gen_loss(self, fake_score):
        if self.loss_type == 'wass':
            return - tf.reduce_mean(fake_score)
        else:
            EPS = 1e-12
            return tf.reduce_mean(-tf.log(fake_score + EPS))

    def get_cycle_loss(self, real, fake_real, caric, fake_caric):
        def rec_loss(orig, rec):
            return tf.reduce_mean(tf.losses.absolute_difference(orig, rec))

        rec_real = self.generator(images=real,
                                  scope='Real2Caric')
        rec_caric = self.generator(images=caric,
                                   scope='Caric2Real')
        fwd_loss = rec_loss(orig=real, rec=rec_real)
        bwd_loss = rec_loss(orig=caric, rec=rec_caric)
        return fwd_loss + bwd_loss

    def get_ucn_loss(self, pos_encs, neg_encs):
        pos_loss = tf.reduce_mean(tf.square(pos_encs[0] - pos_encs[1]))
        neg_diff = tf.reduce_mean(tf.square(neg_encs[0] - neg_encs[1]))
        neg_loss = tf.maximum(0., self.margin - neg_diff)
        return pos_loss + neg_loss

    def build_model(self):

        if self.mode == 'pretrain':
            self.real_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                              'real_faces')
            self.caric_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                               'caric_faces')
            self.real_labels = tf.placeholder(tf.int64, [None],
                                              'real_labels')
            self.caric_labels = tf.placeholder(tf.int64, [None],
                                               'caric_labels')

            '''
            how to form pairs:
                positive: c_base<->c_pos, r_base<->r_pos,
                          c_base<->r_base, c_pos<->r_pos,
                          c_base<->r_pos, c_pos<->r_base
                negative: c_base<->c_neg, r_base<->r_neg,
                          c_neg<->r_neg
            '''
            self.c_base = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                         'caric_base')
            self.c_pos = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                        'caric_pos')
            self.c_neg = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                        'caric_neg')
            self.r_base = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                         'real_base')
            self.r_pos = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                        'real_pos')
            self.r_neg = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                        'real_neg')

            # logits and accuracy
            self.enc_real, _ = self.generator(self.real_images,
                                              scope='Real2Caric')
            self.enc_caric, _ = self.generator(self.caric_images,
                                               scope='Caric2Real')
            enc_c_base, _ = self.generator(self.c_base,
                                           scope='Caric2Real',
                                           reuse=True)
            enc_c_pos, _ = self.generator(self.c_pos,
                                          scope='Caric2Real',
                                          reuse=True)
            enc_c_neg, _ = self.generator(self.c_neg,
                                          scope='Caric2Real',
                                          reuse=True)
            enc_r_base, _ = self.generator(self.r_base,
                                           scope='Real2Caric',
                                           reuse=True)
            enc_r_pos, _ = self.generator(self.r_pos,
                                          scope='Real2Caric',
                                          reuse=True)
            enc_r_neg, _ = self.generator(self.r_neg,
                                          scope="Real2Caric",
                                          reuse=True)

            self.logits_real = self.classifier(encodings=self.enc_real)
            self.logits_caric = self.classifier(encodings=self.enc_caric)

            self.labels = tf.concat([self.real_labels, self.caric_labels], 0)
            self.logits = tf.concat([self.logits_real, self.logits_caric], 0)

            self.pos_pair_one = tf.concat([enc_c_base, enc_r_base, enc_c_pos,
                                           enc_c_base, enc_r_pos, enc_c_pos],
                                          0)
            self.pos_pair_two = tf.concat([enc_c_pos, enc_r_pos, enc_r_pos,
                                           enc_r_base, enc_c_base, enc_r_base],
                                          0)
            self.neg_pair_one = tf.concat([enc_c_base, enc_r_base, enc_r_neg],
                                          0)
            self.neg_pair_two = tf.concat([enc_c_neg, enc_r_neg, enc_c_neg],
                                          0)

            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred,
                                         self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,
                                                   tf.float32))

            # loss and train op
            self.loss_class = \
                tf.losses.sparse_softmax_cross_entropy(self.labels,
                                                       self.logits)
            self.loss_ucn = self.get_ucn_loss(pos_encs=[self.pos_pair_one,
                                                        self.pos_pair_two],
                                              neg_encs=[self.neg_pair_one,
                                                        self.neg_pair_two])

            self.loss = self.loss_class * self.class_weight \
                + self.loss_ucn * self.ucn_weight
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.loss,
                                                          self.optimizer,
                                                          clip_gradient_norm=1)
            # -----------------------------------#

            # summary op
            loss_ucn_summary = tf.summary.scalar('ucn loss',
                                                 self.loss_ucn)
            loss_class_summary = tf.summary.scalar('classification_loss',
                                                   self.loss_class)
            loss_summary = tf.summary.scalar('combined loss',
                                             self.loss)
            accuracy_summary = tf.summary.scalar('accuracy',
                                                 self.accuracy)

            c_base_summ = tf.summary.image('caric bases',
                                           self.c_base)
            r_base_summ = tf.summary.image('real bases',
                                           self.r_base)
            c_pos_summ = tf.summary.image('caric pos', self.c_pos)
            c_neg_summ = tf.summary.image('caric neg', self.c_neg)
            r_pos_summ = tf.summary.image('real pos', self.r_pos)
            r_neg_summ = tf.summary.image('real neg', self.r_neg)

            self.summary_op = tf.summary.merge([
                loss_summary,
                loss_class_summary,
                loss_ucn_summary,
                c_base_summ,
                r_base_summ,
                c_pos_summ,
                c_neg_summ,
                r_pos_summ,
                r_neg_summ,
                accuracy_summary
            ])

        elif self.mode == 'eval':
            self.images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                         'real_faces')

            # source domain
            self.fx = self.content_extractor(self.images)
            self.sampled_images = self.generator(self.fx)

        elif self.mode == 'train':
            self.real_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                              'real_faces')
            self.caric_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
                                               'caric_faces')
            self.real_labels = tf.placeholder(tf.int64, [None],
                                              'real_labels')
            self.caric_labels = tf.placeholder(tf.int64, [None],
                                               'caric_labels')

            # logits and accuracy
            self.enc_real, self.fake_caric = self.generator(self.real_images,
                                                            scope='Real2Caric')
            self.enc_caric, self.fake_real = self.generator(self.caric_images,
                                                            scope='Caric2Real')
            self.logits_real = self.classifier(encodings=self.enc_real)
            self.logits_caric = self.classifier(encodings=self.enc_caric)

            self.labels = tf.concat([self.real_labels, self.caric_labels], 0)
            self.logits = tf.concat([self.logits_real, self.logits_caric], 0)
#---------------------------------------------------------------------------------#
            # self.real_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
            #                                   'real_faces')
            # self.caric_images = tf.placeholder(tf.float32, [None, 64, 64, 3],
            #                                    'caric_faces')
            # self.real_labels = tf.placeholder(tf.int64, [None],
            #                                   'real_labels')
            # self.caric_labels = tf.placeholder(tf.int64, [None],
            #                                    'caric_labels')

            # encodings, transformations and reconstructions
            # self.real_enc, _ = self.encoder(self.real_images,
            #                                 scope_suffix='real')
            # self.caric_enc, self.caric_logits = self.encoder(self.caric_images,
            #                                                  scope_suffix='caric')

            # self.reconst_caric = self.decoder(inputs=self.caric_enc[self.feat_layer - 1],
            #                                   layer=self.feat_layer,
            #                                   scope_suffix='caric')
            # self.trans_real_feat = self.transformer(features=self.real_enc[self.feat_layer - 1],
            #                                         layer=self.feat_layer)
            # self.trans_reconst = self.decoder(inputs=self.trans_real_feat,
            #                                   reuse=True,
            #                                   layer=self.feat_layer,
            #                                   scope_suffix='caric')
            # _, self.reconst_logits = self.encoder(self.trans_reconst,
            #                                       scope_suffix='caric',
            #                                       reuse=True)

            # discriminator scores
            # self.pos_class = self.discriminator(features=self.caric_enc[self.feat_layer - 1],
            #                                     layer=self.feat_layer)
            # self.neg_class = self.discriminator(features=self.trans_real_feat,
            #                                     layer=self.feat_layer,
            #                                     reuse=True)
            # self.reconst_score = self.discriminator(features=self.trans_reconst,
            #                                         layer=0)
            # self.caric_score = self.discriminator(features=self.caric_images,
            #                                       layer=0,
            #                                       reuse=True)

            # accuracy
            # self.pred = tf.argmax(self.reconst_logits, 1)
            # self.correct_pred = tf.equal(self.pred,
            #                              self.real_labels)
            # self.trans_accr = tf.reduce_mean(tf.cast(
            #     self.correct_pred, tf.float32))

            # loss_decoder
            # self.loss_decoder = tf.reduce_mean(tf.losses.absolute_difference(
            #     self.reconst_caric, self.caric_images))

            # classification_loss
            # self.loss_class = \
            #     tf.losses.sparse_softmax_cross_entropy(self.real_labels,
            #                                            self.reconst_logits) \
            #     + tf.losses.sparse_softmax_cross_entropy(self.caric_labels,
            #                                              self.caric_logits)
            # adversarial_loss
            # self.loss_disc = -tf.reduce_mean(self.pos_class
            #                                  - self.neg_class) \
            #     - tf.reduce_mean(self.caric_score -
            #                      self.reconst_score)
            # self.loss_gen = - tf.reduce_mean(self.neg_class) \
            #     - tf.reduce_mean(self.reconst_score)

            # transformer_loss
            # self.loss_transformer = self.loss_gen \
            #     + self.loss_class * self.class_weight

            # optimizer
            # self.dec_opt = tf.train.RMSPropOptimizer(self.learning_rate)
            # self.disc_opt = tf.train.RMSPropOptimizer(self.learning_rate)
            # self.trans_opt = tf.train.RMSPropOptimizer(self.learning_rate)

            # model variables
            # all_vars = tf.trainable_variables()
            # disc_vars = \
            #     [var for var in all_vars if 'discriminator' in var.name]
            # trans_vars = \
            #     [var for var in all_vars if 'transformer' in var.name]
            # dec_vars = \
            #     [var for var in all_vars if 'decoder_caric' in var.name]

            # train op
            # with tf.variable_scope('train_op', reuse=False):
            #     self.disc_op = slim.learning.create_train_op(
            #         self.loss_disc,
            #         self.disc_opt,
            #         variables_to_train=disc_vars,
            #         clip_gradient_norm=0.01)
            #     self.trans_op = slim.learning.create_train_op(
            #         self.loss_transformer,
            #         self.trans_opt,
            #         variables_to_train=trans_vars)
            #     self.dec_op = slim.learning.create_train_op(
            #         self.loss_decoder,
            #         self.dec_opt,
            #         variables_to_train=dec_vars)

            # summary op
            # gen_loss_summary = tf.summary.scalar('gen_loss',
            #                                      self.loss_gen)
            # dec_loss_summary = tf.summary.scalar('loss_dec',
            #                                      self.loss_decoder)
            # accuracy_summary = tf.summary.scalar('trans_accr',
            #                                      self.trans_accr)
            # disc_loss_summary = tf.summary.scalar('disc_loss',
            #                                       self.loss_disc)
            # trans_loss_summary = tf.summary.scalar('transformer_loss',
            #                                        self.loss_transformer)
            # real_images_summary = tf.summary.image('real_images',
            #                                        self.real_images)
            # caric_images_summary = tf.summary.image('caric_images',
            #                                         self.caric_images)
            # trans_reconst_summary = tf.summary.image('trans_reconst',
            #                                          self.trans_reconst)
            # caric_reconst_summary = tf.summary.image('caric_reconst',
            #                                          self.reconst_caric)
            # self.summary_op = tf.summary.merge([gen_loss_summary,
            #                                     dec_loss_summary,
            #                                     accuracy_summary,
            #                                     disc_loss_summary,
            #                                     trans_loss_summary,
            #                                     real_images_summary,
            #                                     caric_images_summary,
            #                                     trans_reconst_summary,
            #                                     caric_reconst_summary])
