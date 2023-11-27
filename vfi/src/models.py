import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import numpy as np
from tensorflow_addons import optimizers as tfa_optimizers
from vfi.src import laploss 
#from basic import warp
from vfi.src import modules
from vfi.src import schedule 
from vfi.src import refine
def _mse_psnr(original, reconstruction,training):
  """Calculates mse and PSNR.

  If training is False, we quantize the pixel values before calculating the
  metrics.

  Args:
    original: Image, in [0, 1].
    reconstruction: Reconstruction, in [0, 1].
    training: Whether we are in training mode.

  Returns:
    Tuple mse, psnr.
  """
  # The images/reconstructions are in [0...1] range, but we scale them to
  # [0...255] before computing the MSE.
  mse_per_batch = tf.reduce_mean(
      tf.math.squared_difference(
          (original * 255.0),
          (reconstruction * 255.0)),
      axis=(1, 2, 3))
  mse = tf.reduce_mean(mse_per_batch)
  psnr_factor = -10. / tf.math.log(10.)
  psnr = tf.reduce_mean(psnr_factor * tf.math.log(mse_per_batch / (255.**2)))
  return mse, psnr

class CompressionSchedule(schedule.KerasSchedule):
  """LR Schedule for compression, with a drop at the end and warmup."""

  def __init__(
      self,
      base_learning_rate,
      num_steps,
      warmup_until = 0.02,
      drop_after = 0.85,
      drop_factor = 0.1,
  ):
    super().__init__(
        base_value=base_learning_rate,
        warmup_steps=int(warmup_until * num_steps),
        vals=[1., drop_factor],
        boundaries=[int(drop_after * num_steps)])


LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule

class _WrapAsWeightDecaySchedule(LearningRateSchedule):
  """Wraps a learning rate schedule into a weight decay schedule."""

  # This class is needed because we want to multiply the weight decay factor
  # for AdamW with the learning rate schedule. This only works for
  # tfa_optimizers.AdamW if we have a subclass of `LearningRateSchedule`.

  def __init__(self, lr_schedule, weight_decay):
    super().__init__()
    self._lr_schedule = lr_schedule
    self._weight_decay = weight_decay

  def __call__(self, step):
    return self._weight_decay * self._lr_schedule(step)
def _make_optimizer_and_lr_schedule(
    schedules_num_steps,
    weight_decay = 0.03,
    learning_rate = 1e-4,
    global_clipnorm = 1.0,
):
  """Returns optimizer and learning rate schedule."""

  lr_schedule = CompressionSchedule(
      base_learning_rate=learning_rate,
      num_steps=schedules_num_steps,
  )
  opt = tfa_optimizers.AdamW(
      learning_rate=lr_schedule,
      # NOTE: We only implement the weight-decay variant where the factor
      # multiplies the LR schedule.
      # Need an instance of LearningRateSchedule, hence the wrap!
      weight_decay=_WrapAsWeightDecaySchedule(lr_schedule, weight_decay),
      global_clipnorm=global_clipnorm,
      beta_1=0.9,
      beta_2=0.98,
      epsilon=1e-9,
  )
  return opt, lr_schedule

    
from vfi.src import config 

feature_extractor_cfg, flow_estimation_cfg = config.MODEL_CONFIG['MODEL_ARCH']

class Model(tf.Module):
    def __init__(self, schedules_num_steps=750_000, **kargs) :
        super(Model, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims']) #2
        self.feature_estimation = modules.feature_extractor(**feature_extractor_cfg)
        self.block = (      [modules.Head( 
                            in_planes=2*(kargs['motion_dims'][-1-i] * kargs['depths'][-1-i] + kargs['embed_dims'][-1-i]), # 2 mf ,af
                            scale=kargs['scales'][-1-i], #16-8-4
                            c=kargs['hidden_dims'][-1-i],
                            in_else=6 if i==0 else 17) 
                            for i in range(self.flow_num_stage)]) # 0 1 
        self.refine = refine.refine(kargs['c'] * 2)

        self._all_trainable_variables = None
        #self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate=.0001, decay_steps=750000, decay_rate=0.464158)
        #self.learning_rate_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=0.0001, decay_steps=750000)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.learning_rate_schedule)
        #self.optimizer = tfa_optimizers.AdamW(learning_rate= self.learning_rate_schedule,weight_decay=0.0001)
        self._optimizer, self._learning_rate_schedule = (_make_optimizer_and_lr_schedule(schedules_num_steps))

    def warp_features(self, af, flow):
        y0 = []
        y1 = []
        B = af[0].shape[0] // 2
        for x in af:
            y0.append(modules.warp(x[:B], flow[..., 0:2]))
            y1.append(modules.warp(x[B:], flow[...,2:4]))
            flow=tf.image.resize(flow, [  flow.shape[1] //2  , flow.shape[2]//2 ] ) *0.5
        return y0, y1

   
    def calculate_flow(self, img0,img1, timestep, af=None, mf=None):
        B = img0.size(0)
        flow, mask = None, None
        # appearence_features & motion_features
        if (af is None) or (mf is None):
            af, mf = self.feature_estimation(img0, img1)
        for i in range(self.flow_num_stage):
            t = tf.fill( list(mf[-1-i][:B].shape), timestep)
            if flow != None:
                warped_img0 = modules.warp(img0, flow[..., :2])
                warped_img1 = modules.warp(img1, flow[..., 2:4])
                flow_, mask_ = self.block[i](
                    tf.concat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],-1),
                    tf.concat([img0, img1, warped_img0, warped_img1, mask], -1),
                    flow
                    )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.sblock[i](
                    tf.concat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],-1),
                    tf.concat([img0, img1], -1),
                    None
                    )

        return flow, mask

    def coraseWarp_and_Refine(self, img0,img1, af, flow, mask):
        warped_img0 = modules.warp(img0, flow[...,:2])
        warped_img1 = modules.warp(img1, flow[...,2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.refine(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = tf.keras.activations.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = tf.clip_by_value(merged + res, 0, 1)
        return pred

    def _iter_trainable_variables(self):

        def ensure_nonempty(seq):
            if not seq:
                raise ValueError("No trainable variables!")
            return seq

        yield from ensure_nonempty(self.feature_estimation.trainable_variables)
        yield from ensure_nonempty(self.refine.trainable_variables)
        yield from ensure_nonempty(self.block.trainable_variables)

    @property
    def all_trainable_variables(self):
        if self._all_trainable_variables is None:
            self._all_trainable_variables = list(self._iter_trainable_variables())
            assert self._all_trainable_variables
            assert len(self._all_trainable_variables) == len(self.trainable_variables)
        return self._all_trainable_variables
            
    def train_step(self, img0,gt,img1):

        with tf.GradientTape() as tape:
            pred,metric = self.interpolator(img0,gt,img1)
            loss= metric['loss']
        var_list = self.all_trainable_variables
        gradients = tape.gradient ( loss, var_list)
        self._optimizer.apply_gradients(zip(gradients,var_list))

        return metric
    def write_ckpt(self, path, step):
        """Creates a checkpoint at `path` for `step`."""
        print('Creates a checkpoint at'+ str(path) + 'for' + str(step))
        ckpt = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
        manager.save(checkpoint_number=step)
        return tf.train.latest_checkpoint(path)
        
    def load_ckpt(self, path):
        """load a checkpoint at `path` for `step`."""
        ckpt = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
        print('load a checkpoint at'+ str(path) + 'for' + manager.latest_checkpoint)
        return ckpt.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    @property
    def global_step(self):
        """Returns the global step variable."""
        return self._optimizer.iterations
    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def interpolator(self,img0,gt,img1, timestep=0.5):
        B = img0.shape[0] # batch size 
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        
        # appearence_features & motion_features
        af, mf  = self.feature_estimation(img0, img1) # feature extractor [[2,256,30,60],,...]5개, [[2,256,30,60],...] 3개 
#################################################################

        for i in range(self.flow_num_stage): # i=0,1

            t = tf.fill( list(mf[-1-i][:B].shape) , timestep)
            
            if flow != None:
                m=tf.concat([t*mf[-1-i][:B], (1-timestep)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],-1)
                x2=tf.concat((img0, img1, warped_img0, warped_img1, mask), -1)
                
                flow_d, mask_d = self.block[i]( motion_feature=m, 
                                                x=x2, 
                                                flow=flow) # 3 3 3 3 1 
                flow = flow + flow_d 
                mask = mask + mask_d
            else:
                #    def forward(self, motion_feature, x, flow): # /16 /8 /4
                m=tf.concat([t*mf[-1-i][:B], (1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]], -1)
                x2=tf.concat((img0, img1), -1)
            
                flow, mask = self.block[i]( 
                                            motion_feature=m, 
                                            x=x2, 
                                            flow=None) # 1,6,480,960

            mask_list.append(tf.keras.activations.sigmoid(mask))
            flow_list.append(flow)

            warped_img0 = modules.warp(img0, flow[..., :2])
            warped_img1 = modules.warp(img1, flow[..., 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        
        c0, c1 = self.warp_features(af, flow)
        tmp = self.refine(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[..., :3] * 2 - 1
        pred = tf.clip_by_value(merged[-1] + res, 0, 1)

        #loss_l1 = tf.reduce_mean(tf.abs(pred-gt))
        loss_l1 = tf.reduce_mean(laploss.laploss(pred,gt))
        for merge in merged:
            loss_l1 += tf.reduce_mean(laploss.laploss(merge, gt)) * 0.5
            #loss_l1 += tf.reduce_mean(tf.abs(merge-gt)) * 0.5
        mse,psnr = _mse_psnr(gt,pred,False)
        metric={'loss' : loss_l1 , 'psnr' : psnr , 'mse':mse }     
        #return flow_list, mask_list, merged, pred
    
        return pred, metric 
    

