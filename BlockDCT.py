# -*- coding: utf-8 -*
''' Block Discrete Cosine Transform (block DCT) using TensorFlow'''
#  Copyright 2018 Takamichi Miyata
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import cv2
import tensorflow as tf

#Definition of functions
def conv2d_block(x, W):
    with tf.name_scope("conv2d_block"):
        #                               strideを[8,8]に設定
        return tf.nn.conv2d(x, W, strides=[1, 8, 8, 1], padding='VALID')

def conv2d_deblock(x, W):
    with tf.name_scope("conv2d_deblock"):
        r=8
        X = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        s = tf.shape(X)
        X = tf.reshape(X, (1, s[1], s[2], 8,8))
        X = tf.transpose(X, [0, 1, 3, 2, 4])  # bsize, a, b, 1, 1
        return tf.reshape(X, (s[0],s[1]*8, s[2]*8, 1))

def make_DCT_filter():
    with tf.name_scope("DCT"):
        # Initialize DCT filters
        DCT_filter_n = np.zeros([8, 8, 1, 64])
        # Definition of 8x8 mesh grid
        XX, YY = np.meshgrid(range(8), range(8))
        # DCT basis as filters
        C=np.ones(8)
        C[0]=1/np.sqrt(2)
        for v in range(8):
            for u in range(8):
                DCT_filter_n[:, :, 0, u+v*8]=(2*C[v]*C[u]/8)*np.cos((2*YY+1)*v*np.pi/(16))*np.cos((2*XX+1)*u*np.pi/(16))

        DCT_filter=tf.constant(DCT_filter_n.astype(np.float32))

        return DCT_filter

def make_IDCT_filter():
    with tf.name_scope("IDCT"):
        # Initialize inverse DCT (IDCT) filters
        IDCT_filter_n = np.zeros([1, 1, 64, 64])
        # IDCT basis as filters
        C=np.ones(8)
        C[0]=1/np.sqrt(2)
        for v in range(8):
            for u in range(8):
                for j in range(8):
                    for i in range(8):
                        IDCT_filter_n[0,0,u+v*8,i+j*8]=(2*C[v]*C[u]/8)*np.cos((2*j+1)*v*np.pi/(16))*np.cos((2*i+1)*u*np.pi/(16))

        IDCT_filter=tf.constant(IDCT_filter_n.astype(np.float32))

        return IDCT_filter

def main():
    # Preparation DCT filter coefficients
    DCT_filter = make_DCT_filter()
    IDCT_filter = make_IDCT_filter()

    # Placeholder for the input image
    inp_im_p = tf.placeholder(tf.float32, shape=[None, None])
    inp_im_b = tf.expand_dims(tf.expand_dims(inp_im_p, 0), 3)
    # OPs
    out_coef = conv2d_block(inp_im_b,  DCT_filter)
    out_im = conv2d_deblock(out_coef, IDCT_filter)


    # Reading the input image
    img = cv2.imread('./cameraman.png', cv2.IMREAD_GRAYSCALE)
    inp_im_n = img.astype(np.float32)

    # Defining summary for TensorBoard
    _ = tf.summary.image('input', inp_im_b, max_outputs=1, collections=None)
    _ = tf.summary.image('DCT_basis', tf.transpose(DCT_filter, [3,0,1,2]), max_outputs=64, collections=None)
    _ = tf.summary.image('DCT_coef', tf.transpose(out_coef, [3,1,2,0]), max_outputs=64, collections=None)
    _ = tf.summary.image('ouput', out_im, max_outputs=1, collections=None)

    merged_summary = tf.summary.merge_all()

    # Preparation for TensorBoard
    log_dir = './log'

    # Starting the TF session
    with tf.Session() as sess:
        summary = sess.run(merged_summary, feed_dict={inp_im_p: inp_im_n})

        # Output for TensorBoard
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary_writer.add_summary(summary)
        summary_writer.close()

if __name__ == '__main__':
    main()
