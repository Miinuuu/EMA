# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for models."""

import tensorflow as tf
from vfi.src import models 
from vfi.src import config 
from vfi.src import vimeo_datasets 
import argparse
from tqdm import tqdm
import wandb as wandb
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


feature_extractor_cfg, flow_estimation_cfg = config.MODEL_CONFIG['MODEL_ARCH']
model= models.Model( **flow_estimation_cfg)

class ModelsTest(tf.test.TestCase):

  def eval(self,model) :
    test_step = tf.function(model.interpolator)
    base_shape = (1, 256, 256 , 3)
    
    img0 = tf.random.stateless_normal(base_shape, seed=[1, 1])
    gt = tf.random.stateless_normal(base_shape, seed=[1, 1])
    img1 = tf.random.stateless_normal(base_shape, seed=[1, 1])

    _, metrics = test_step(img0,gt,img1,0.5)


  def _restore_evaluate(self, ckpt_p=None):
    """Restore and evaluate a model for the given stage."""
    print("""Restore and evaluate a model for the given stage.""")
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(ckpt_p).assert_existing_objects_matched()
      #ckpt.restore(ckpt_p).expect_partial()
    self.eval(model)
    

  def test_train_eval(self):
    ckpt_dir = self.create_tempdir().full_path

    base_shape = (1 , 256, 256 , 3)
    train_step = tf.function(model.train_step)

    img0 = tf.random.stateless_normal(base_shape, seed=[1, 1])
    gt = tf.random.stateless_normal(base_shape, seed=[1, 1])
    img1 = tf.random.stateless_normal(base_shape, seed=[1, 1])
    
    metrics=train_step(img0,gt,img1)
            
    ckpt_p = model.write_ckpt(ckpt_dir, step=0)
    self._restore_evaluate(ckpt_p)

if __name__ == "__main__":
  tf.test.main()
