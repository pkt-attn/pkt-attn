import torch
import argparse
import numpy as np

from load_data_ggnn import DATA_ggnn
from run import train, test
from model import MODEL
import random
import os


# set random seed
def seed_torch(seed=0):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(fold):

    parser = argparse.ArgumentParser()
    # 可设置参数
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--EPOCH', type=int, default=300)
    parser.add_argument('--fold', type=int, default=fold)
    parser.add_argument('--dataset', type=str, default='codeforces')
    parser.add_argument('--set_seed', type=bool, default=True)
    parser.add_argument('--node_embed_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--init_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--concept_embed_dim', type=int, default=128)
    parser.add_argument('--num_nodes', type=int, default=200)  # <=200


    root = '../../data/' + parser.parse_args().dataset + '/ggnn/'
    train_path = root + 'ggnn_train' + str(parser.parse_args().fold)
    val_path = root + 'ggnn_valid' + str(parser.parse_args().fold)
    test_path = root + 'aggnn_test'

    if parser.parse_args().dataset == 'codeforces':

        # PKT parameters
        parser.add_argument('--num_concepts', type=int, default=37)  # 实际数量
        parser.add_argument('--num_problems', type=int, default=7152)
        parser.add_argument('--seqlen', type=int, default=200)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--hidden_layers', type=int, default=2)

        # ggnn params
        parser.add_argument('--vocablen', type=int, default=115730)

    #  ________________________________________________________________________________________________________

    params = parser.parse_args()
    print(params)

    # set random seed
    if params.set_seed is True:
        seed_torch(0)

    # load data
    data = DATA_ggnn(num_concepts=params.num_concepts, seqlen=params.seqlen)

    print(train_path)
    train_p_id, train_c_id, train_node_id, train_edge, train_edge_type, train_target_c, train_result, train_c_embed, \
    train_x_result = data.load_data(train_path)
    print(val_path)
    val_p_id, val_c_id, val_node_id, val_edge, val_edge_type, val_target_c, val_result, val_c_embed, \
    val_x_result = data.load_data(val_path)

    model = MODEL(num_concepts=params.num_concepts,
                  num_problems=params.num_problems,
                  hidden_dim=params.hidden_dim,
                  hidden_layers=params.hidden_layers,
                  concept_embed_dim=params.concept_embed_dim,
                  vocablen=params.vocablen,
                  node_embed_dim=params.node_embed_dim,
                  num_nodes=params.num_nodes,
                  gpu=params.gpu)

    model.init_params()
    model.init_embeddings()


    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.init_lr, weight_decay=params.weight_decay)


    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()

    best_valid_auc = 0
    count = 0
    for idx in range(params.EPOCH):
        train_loss, train_accuracy, train_auc = train(model, params, optimizer, train_p_id, train_c_id,
                train_node_id, train_edge, train_edge_type, train_target_c, train_result, train_c_embed, train_x_result)
        print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (
            idx + 1, params.EPOCH, train_loss, train_auc, train_accuracy))

        valid_loss, valid_accuracy, valid_auc = test(model, params, val_p_id, val_c_id, val_node_id, val_edge,
                val_edge_type, val_target_c, val_result, val_c_embed, val_x_result)
        print('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' % (
            idx + 1, params.EPOCH, valid_auc, valid_accuracy))

        if valid_auc > best_valid_auc:
            count = 0
            print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
            best_valid_auc = valid_auc
            best_train_auc = train_auc
            best_model = model

        else:
            count += 1
            if params.dataset == 'codeforces':
                if count == 5:
                    break
            elif params.dataset == 'oj':
                if count == 10:
                    break


    # test
    test_p_id, test_c_id, test_node_id, test_edge, test_edge_type, test_target_c, test_result, test_c_embed, \
    test_x_result = data.load_data(test_path)


    test_loss, test_accuracy, test_auc = test(best_model, params, test_p_id, test_c_id, test_node_id, test_edge,
            test_edge_type, test_target_c, test_result, test_c_embed, test_x_result)
    print('test auc : %3.5f, test accuracy : %3.5f' % (test_auc, test_accuracy))

    return best_train_auc, best_valid_auc, test_auc


if __name__ == '__main__':
    bt1, bv1, test_auc_1 = main(1)
    bt2, bv2, test_auc_2 = main(2)
    bt3, bv3, test_auc_3 = main(3)
    bt4, bv4, test_auc_4 = main(4)
    bt5, bv5, test_auc_5 = main(5)
    print('best train auc = ', bt1, ', best valid auc1 = ', bv1, ', test auc1 = ', test_auc_1)
    print('best train auc = ', bt2, ', best valid auc2 = ', bv2, ', test auc2 = ', test_auc_2)
    print('best train auc = ', bt3, ', best valid auc3 = ', bv3, ', test auc3 = ', test_auc_3)
    print('best train auc = ', bt4, ', best valid auc4 = ', bv4, ', test auc4 = ', test_auc_4)
    print('best train auc = ', bt5, ', best valid auc5 = ', bv5, ', test auc5 = ', test_auc_5)
    avg = (test_auc_1 + test_auc_2 + test_auc_3 + test_auc_4 + test_auc_5) / 5.0
    var = (pow(test_auc_1 - avg, 2) + pow(test_auc_2 - avg, 2) + pow(test_auc_3 - avg, 2) +
           pow(test_auc_4 - avg, 2) + pow(test_auc_5 - avg, 2)) / 5.0
    print('average test auc = ', avg)
    print('var = ', var)
    print('ggnn')
