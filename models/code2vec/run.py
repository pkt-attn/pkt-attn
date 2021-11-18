import math
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

import utils as utils

def train(model, params, optimizer, p_id, c_id, source_token, path_token, target_token, context_mask, \
     target_c, result, c_embed, x_result):

    N = int(math.floor(len(p_id) / params.batch_size))

    # shuffle data
    shuffle_index = np.random.permutation(p_id.shape[0])
    p_id = p_id[shuffle_index]
    c_id = c_id[shuffle_index]
    source_token = source_token[shuffle_index]
    path_token = path_token[shuffle_index]
    target_token = target_token[shuffle_index]
    context_mask = context_mask[shuffle_index]
    target_c = target_c[shuffle_index]
    result = result[shuffle_index]
    c_embed = c_embed[shuffle_index]
    x_result = x_result[shuffle_index]

    pred_list = []
    target_list = []
    model.train()
    epoch_loss = 0

    # for idx in tqdm(range(N)):
    for idx in range(N):
        p_id_seq = p_id[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        c_id_seq = c_id[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        source_token_seq = source_token[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        path_token_seq = path_token[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        target_token_seq = target_token[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        context_mask_seq = context_mask[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        target_c_seq = target_c[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        result_seq = result[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        c_embed_seq = c_embed[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        x_result_seq = x_result[idx * params.batch_size: (idx + 1) * params.batch_size, :]

        input_p_id = utils.variable(torch.LongTensor(p_id_seq), params.gpu)
        input_c_id = utils.variable(torch.FloatTensor(c_id_seq), params.gpu)
        input_source_token = utils.variable(torch.LongTensor(source_token_seq), params.gpu)
        input_path_token = utils.variable(torch.LongTensor(path_token_seq), params.gpu)
        input_target_token = utils.variable(torch.LongTensor(target_token_seq), params.gpu)
        input_context_mask = utils.variable(torch.FloatTensor(context_mask_seq), params.gpu)
        input_target_c = utils.variable(torch.FloatTensor(target_c_seq), params.gpu)
        input_result = utils.variable(torch.FloatTensor(result_seq), params.gpu).view(-1, 1)
        input_c_embed = utils.variable(torch.FloatTensor(c_embed_seq), params.gpu)
        input_x_result = utils.variable(torch.FloatTensor(x_result_seq), params.gpu)

        model.zero_grad()
        loss, filtered_pred, filtered_target = model(input_p_id, input_c_id, input_source_token,
                                    input_path_token, input_target_token, input_context_mask,
                                    input_target_c, input_result, input_c_embed, input_x_result)
        loss.backward()

        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc


def test(model, params, p_id, c_id, source_token, path_token, target_token, context_mask, \
     target_c, result, c_embed, x_result):

    N = int(math.floor(len(p_id) / params.batch_size))

    pred_list = []
    target_list = []
    model.eval()
    epoch_loss = 0

    # for idx in tqdm(range(N)):
    for idx in range(N):
        p_id_seq = p_id[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        c_id_seq = c_id[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        source_token_seq = source_token[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        path_token_seq = path_token[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        target_token_seq = target_token[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        context_mask_seq = context_mask[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        target_c_seq = target_c[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        result_seq = result[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        c_embed_seq = c_embed[idx * params.batch_size: (idx + 1) * params.batch_size, :]
        x_result_seq = x_result[idx * params.batch_size: (idx + 1) * params.batch_size, :]

        input_p_id = utils.variable(torch.LongTensor(p_id_seq), params.gpu)
        input_c_id = utils.variable(torch.FloatTensor(c_id_seq), params.gpu)
        input_source_token = utils.variable(torch.LongTensor(source_token_seq), params.gpu)
        input_path_token = utils.variable(torch.LongTensor(path_token_seq), params.gpu)
        input_target_token = utils.variable(torch.LongTensor(target_token_seq), params.gpu)
        input_context_mask = utils.variable(torch.FloatTensor(context_mask_seq), params.gpu)
        input_target_c = utils.variable(torch.FloatTensor(target_c_seq), params.gpu)
        input_result = utils.variable(torch.FloatTensor(result_seq), params.gpu).view(-1, 1)
        input_c_embed = utils.variable(torch.FloatTensor(c_embed_seq), params.gpu)
        input_x_result = utils.variable(torch.FloatTensor(x_result_seq), params.gpu)


        loss, filtered_pred, filtered_target = model(input_p_id, input_c_id, input_source_token,
                                    input_path_token, input_target_token, input_context_mask,
                                    input_target_c, input_result, input_c_embed, input_x_result)
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc