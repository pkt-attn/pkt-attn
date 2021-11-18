import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from utils import *

class MODEL(nn.Module):
    def __init__(self, num_concepts, num_problems, hidden_dim, hidden_layers, nodes_dim,
                 paths_dim, codevec_size, concept_embed_dim, np, gpu):
        super(MODEL, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.gpu = gpu
        self.num_concepts = num_concepts
        self.codevec_size = codevec_size
        self.np = np
        self.concept_embed_dim = concept_embed_dim

        self.dropout = nn.Dropout(0.6)

        # DKT network
        self.p_id_embed = nn.Embedding(num_problems + 1, concept_embed_dim)
        self.c_id_embed = nn.Parameter(torch.empty(2 * num_concepts + 1, concept_embed_dim), requires_grad=True)

        # input feature dim
        self.LSTM_feature_dim = 2 + num_concepts + 1

        # code2vec network
        self.node_embedding = nn.Embedding(nodes_dim, codevec_size)
        self.path_embedding = nn.Embedding(paths_dim, codevec_size)
        self.cv_fc = nn.Linear(3 * codevec_size, codevec_size, bias=False)
        a = torch.nn.init.uniform_(torch.empty(codevec_size, 1, dtype=torch.float32, requires_grad=True))
        self.a = nn.parameter.Parameter(a, requires_grad=True)

        self.LSTM = nn.LSTM(input_size=self.LSTM_feature_dim, hidden_size=hidden_dim,
                            num_layers=hidden_layers, batch_first=True)
        self.predict_Linear = nn.Linear(hidden_dim, num_concepts + 1, bias=True)


    def init_embeddings(self):
        nn.init.kaiming_normal_(self.node_embedding.weight)
        nn.init.kaiming_normal_(self.path_embedding.weight)
        nn.init.kaiming_normal_(self.cv_fc.weight)


    def init_params(self):
        nn.init.kaiming_normal_(self.predict_Linear.weight)
        nn.init.constant_(self.predict_Linear.bias, 0)
        nn.init.kaiming_normal_(self.c_id_embed.data)
        nn.init.kaiming_normal_(self.p_id_embed.weight)


    def code2vec(self, paths, context_embedded, bs, seqlen):

        num_paths = paths.shape[2]

        # Fully Connected
        context_after_dense = variable(torch.tanh(context_embedded), self.gpu)

        # Attention weight
        attention_weight = self.a.repeat(bs, seqlen, 1, 1)  # attention_weight = [bs, code_vector_size, 1]
        attention_weight = torch.bmm(context_after_dense.view(bs * seqlen, num_paths, self.codevec_size),
                                     attention_weight.view(bs * seqlen, self.codevec_size, 1)).view(bs, seqlen,
                                                                                                    num_paths, 1)
        # code_vectors = [bs, code_vector_size]
        code_vectors = torch.sum(torch.mul(context_after_dense, attention_weight.expand_as(context_after_dense)), dim=2)

        return code_vectors

    def preprocess(self, paths, starts, ends):
        # 取前n个path
        if self.np < 200:
            sample = random.sample(range(0, 200), self.np)
            paths = paths[:, :, sample]
            starts = starts[:, :, sample]
            ends = ends[:, :, sample]

        # Embedding
        starts_embedded = self.node_embedding(starts)
        paths_embedded = self.path_embedding(paths)
        ends_embedded = self.node_embedding(ends)

        # Concatenate
        context_embedded = torch.cat((starts_embedded, paths_embedded, ends_embedded), dim=3)
        context_embedded = self.dropout(context_embedded)
        context_embedded = self.cv_fc(context_embedded)

        return context_embedded, starts, paths, ends, paths_embedded


    def forward(self, p_id, c_id, starts, paths, ends, masks, target_c, result, c_embed, cur_result):
        bs = p_id.shape[0]
        seqlen = p_id.shape[1]

        #####################################################################################################
        # models
        # p_id_embed = self.p_id_embed(p_id)

        LSTM_input = torch.cat([c_embed, cur_result], 2)

        # LSTM
        LSTM_out, final_status = self.LSTM(LSTM_input)
        LSTM_out = LSTM_out.contiguous()

        #######################################################################################################
        # prediction

        # 多知识点，除以知识点个数
        num_concepts = torch.sum(target_c, 2).view(-1)

        prediction = self.predict_Linear(LSTM_out.view(bs * seqlen, -1))  # 去掉dropout
        prediction_1d = torch.bmm(prediction.unsqueeze(1),
                                  target_c.view(bs * seqlen, -1).unsqueeze(2)).squeeze(2)
        mask = num_concepts.gt(0)
        num_concepts = torch.masked_select(num_concepts, mask)
        filtered_pred = torch.masked_select(prediction_1d.squeeze(1), mask)
        filtered_pred = torch.div(filtered_pred, num_concepts)
        filtered_target = torch.masked_select(result.squeeze(1), mask)
        loss = F.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target


