#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import deepchem as dc
import dgllife
import dgl
import torch
import numpy as np
import pandas as pd
from deepchem.models.torch_models import MPNNModel
import json
import tensorflow as tf

tf.random.set_seed(123)
import deepchem as dc
from deepchem.molnet import load_tox21

import torch.profiler

from deepchem.models.optimizers import Adam

from deepchem.feat.graph_data import GraphData
from deepchem.models.losses import Loss
from deepchem.utils.typing import ArrayLike, LossFn, OneOrMany
from deepchem.models.losses import SparseSoftmaxCrossEntropy
import logging
import time

try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection


#import multiprocessing as mp
import torch.multiprocessing as mp


# In[2]:
def PrepareBatch(conn, conn2, conn3, dataset):
    #print("PrepareBatch starts")
    model = MPNNModel(
        mode='classification',
        n_tasks=12,
        batch_size=128,
        num_step_message_passing=3,
        num_step_set2set=3,
        num_lyaer_set2set=8,
        optimizer=Adam(learning_rate=0.0045),
        tensorboard=False,
        model_dir='models',
        number_atom_features= 30 #30~33 depends on optional features
        )

    deterministic = False
    batches = model.default_generator(dataset, epochs = 10, deterministic = deterministic)



    for batch in batches:
        input_tensors, label_tensors, weight_tensors = model._prepare_batch(batch)
        conn.send(input_tensors)
        conn2.send(label_tensors)
        conn3.send(weight_tensors)

    final_msg = "END"
    conn.send(final_msg)
    conn.close()
    conn2.close()
    conn3.close()
    #print(f"Prepare batch is ended, executed {PrepareBatchCounter} times.")

def TrainModel(conn, conn2, conn3, train_dataset, valid_dataset, test_dataset):
    model = MPNNModel(
        mode='classification',
        n_tasks=12,
        batch_size=128,
        num_step_message_passing=3,
        num_step_set2set=3,
        num_lyaer_set2set=8,
        optimizer=Adam(learning_rate=0.0045),
        tensorboard=False,
        model_dir='models',
        number_atom_features= 30 #30~33 depends on optional features
        )

    variables = None
    callbacks =[]
    checkpoint_interval = 1000
    max_checkpoints_to_keep = 5
    logger = logging.getLogger(__name__)

    #fit_generator
    if not isinstance(callbacks, SequenceCollection):
        callbacks = [callbacks]
    model._ensure_built()
    model.model.train()
    avg_loss = 0.0
    last_avg_loss = 0.0
    averaged_batches = 0
    if model.loss is None:
        loss = model._loss_fn
    if variables is None:
        optimizer = model._pytorch_optimizer
        lr_schedule = model._lr_schedule
    else:
        var_key = tuple(variables)
        if var_key in model._optimizer_for_vars:
            optimizer, lr_schedule = model._optimizer_for_vars[var_key]
        else:
            optimizer = model.optimizer._create_pytorch_optimizer(variables)
            if isinstance(model.optimizer.learning_rate, LearningRateSchedule):
                lr_schedule = model.optimizer.learning_rate._create_pytorch_schedule(optimizer)
            else:
                lr_schedule = None
            model._oprimizer_for_vars[var_key] = (optimizer, lr_schedule)
    time1 = time.time()

    #TrainModelCounter = 0
    
    #Main training loop
    while 1:
        #TrainModelCounter += 1
        inputs = conn.recv()
        if inputs == "END":
            #print(f"Train Model is ended, executed {TrainModelCounter} times.")
            break
        labels = conn2.recv()
        weights = conn3.recv()

        #Execute the loss funtion, accumulating the gradients.
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]

        optimizer.zero_grad()
        outputs = model.model(inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        if model._loss_outputs is not None:
            outputs = [outputs[i] for i in model._loss_outputs]

        batch_loss = model._loss_fn(outputs, labels, weights) 
        batch_loss.backward()
        optimizer.step()
        if lr_schedule is not None:
            lr_schedule.step()
        model._global_step += 1
        current_step = model._global_step

        avg_loss += batch_loss

        #Report progress and write checkpoints.
        averaged_batches +=1
        should_log = (current_step % model.log_frequency ==0)
        if should_log:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info(
                'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
        last_avg_loss = avg_loss
        avg_loss = 0.0
        averaged_batches = 0

        if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval -1 :
            model.save_checkpoint(max_checkpoints_to_keep) 
        for c in callbacks:
            c(current_step)
        if model.tensorboard and should_log:
            model._log_scalar_to_tensorboard('loss', batch_loss, current_step)
        if (model.wandb_logger is not None) and should_log:
            all_data = dict({'train/loss': batch_loss})
            model.wandb_logger.log_data(all_data, step=current_step)

    #report final results
    if averaged_batches > 0:
        avg_loss = float(avg_loss) / averaged_batches
        logger.info(
                'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
        print(f"Last avg loss is {last_avg_loss}")
        last_avg_loss = avg_loss
    if checkpoint_interval > 0:
        model.save_checkpoint(max_checkpoints_to_keep)

    time2 = time.time()
    logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
    print(f"last avg loss is : {last_avg_loss}")
   
    '''
    #Evaluate Model
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score,
            np.mean,
            mode="classification")

    training_score = model.evaluate(train_dataset, [metric], transformers)
    validation_score = model.evaluate(valid_dataset, [metric], transformers)
    test_score = model.evaluate(test_dataset, [metric], transformers)
    print(f"Training score : {training_score} \n Validation score: {validation_score} \n Test score : {test_score}")
    '''

# In[3]:


if __name__=='__main__':
    mp.set_sharing_strategy('file_system')
    #preparing dataset
    tox21_tasks, tox21_datasets, transformers = load_tox21(
            featurizer = dc.feat.MolGraphConvFeaturizer(use_edges= True, use_chirality = False, use_partial_charge = False), 
            splitter='random'
            )

    train_dataset, valid_dataset, test_dataset = tox21_datasets

    #preparing multiprocessing with Pipe
    ctx = mp.get_context('fork')

    parent_conn, child_conn = ctx.Pipe()
    parent_conn2, child_conn2 = ctx.Pipe()
    parent_conn3, child_conn3 = ctx.Pipe()

    #Start training with task parallelism
    start = time.time()

    p1 = ctx.Process(target=PrepareBatch, args=(parent_conn, parent_conn2, parent_conn3, train_dataset))
    p2 = ctx.Process(target=TrainModel, args=(child_conn, child_conn2, child_conn3, train_dataset, valid_dataset, test_dataset))
    
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    end = time.time()
    print("total trining time is:",end-start, "sec")
