import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.LSTM_feature_dim = 2 + concept_embed_dim + num_concepts + 1

        # code2vec network
        self.node_embedding = nn.Embedding(nodes_dim, codevec_size)
        self.path_embedding = nn.Embedding(paths_dim, codevec_size)
        self.cv_fc = nn.Linear(3 * codevec_size, codevec_size, bias=False)
        a = torch.nn.init.uniform_(torch.empty(codevec_size, 1, dtype=torch.float32, requires_grad=True))
        self.a = nn.parameter.Parameter(a, requires_grad=True)

        self.LSTM = nn.LSTM(input_size=self.LSTM_feature_dim, hidden_size=hidden_dim,
                            num_layers=hidden_layers, batch_first=True)
        self.predict_Linear = nn.Linear(hidden_dim, num_concepts + 1, bias=True)

        self.concept_embedding = nn.Parameter(torch.empty(num_concepts + 1, concept_embed_dim), requires_grad=True)
        self.transdim = nn.Linear(codevec_size, concept_embed_dim, bias=False)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.node_embedding.weight)
        nn.init.kaiming_normal_(self.path_embedding.weight)
        nn.init.kaiming_normal_(self.cv_fc.weight)
        nn.init.kaiming_normal_(self.concept_embedding.data)


    def init_params(self):
        nn.init.kaiming_normal_(self.predict_Linear.weight)
        nn.init.constant_(self.predict_Linear.bias, 0)
        nn.init.kaiming_normal_(self.c_id_embed.data)
        nn.init.kaiming_normal_(self.p_id_embed.weight)
        nn.init.kaiming_normal_(self.transdim.weight)

    def preprocess(self, paths, starts, ends):

        # 取n个path
        if self.np < 200:
            paths = paths[:, :, :self.np]
            starts = starts[:, :, :self.np]
            ends = ends[:, :, :self.np]

        # Embedding
        starts_embedded = self.node_embedding(starts)
        paths_embedded = self.path_embedding(paths)
        ends_embedded = self.node_embedding(ends)

        # Concatenate
        context_embedded = torch.cat((starts_embedded, paths_embedded, ends_embedded), dim=3)
        context_embedded = self.dropout(context_embedded)
        context_embedded = self.cv_fc(context_embedded)

        return context_embedded, starts, paths, ends

    def attention(self, c_id, paths, c_embed, embedded, bs, seqlen):

        if(embedded.shape[3] != self.concept_embed_dim):
            embedded = self.transdim(embedded)

        num_paths = paths.shape[2]

        # interactive attention
        c_embed = c_embed.unsqueeze(3).expand(bs, seqlen, self.num_concepts + 1, self.concept_embed_dim)

        # the query is concept
        # 习题知识点个数，用于c_p_attn的AvgPooling
        num = torch.sum(c_id, dim=2).unsqueeze(2)
        num = torch.masked_fill(num, num.eq(0), 1)
        concept_query = c_embed * self.concept_embedding.repeat(bs, seqlen, 1, 1)
        c_p_attn_weight = torch.bmm(concept_query.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim),
                                    embedded.view(bs * seqlen, num_paths,
                                                  self.concept_embed_dim).permute(0, 2, 1))
        c_p_attn_weight = torch.softmax(c_p_attn_weight, dim=2)
        c_p_attn_out = torch.bmm(c_p_attn_weight, embedded.view(bs * seqlen, num_paths, self.concept_embed_dim))
        c_p_attn_out = c_p_attn_out * c_embed.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim)
        c_p_attn_out = torch.sum(c_p_attn_out, dim=1) / num.view(-1).unsqueeze(1)
        c_p_attn_out = c_p_attn_out.view(bs, seqlen, self.concept_embed_dim)

        return c_p_attn_out

    def attention_all(self, c_id, paths, c_embed, embedded, bs, seqlen):

        if(embedded.shape[3] != self.concept_embed_dim):
            embedded = self.transdim(embedded)

        num_paths = paths.shape[2]

        # interactive attention
        # c_embed = c_embed.unsqueeze(3).expand(bs, seqlen, self.num_concepts + 1, self.concept_embed_dim)
        c_embed = variable(torch.ones(bs, seqlen, self.num_concepts + 1, self.concept_embed_dim), self.gpu)

        # the query is concept
        # 习题知识点个数，用于c_p_attn的AvgPooling
        # num = torch.sum(c_id, dim=2).unsqueeze(2)
        # num = torch.masked_fill(num, num.eq(0), 1)
        concept_query = c_embed * self.concept_embedding.repeat(bs, seqlen, 1, 1)
        c_p_attn_weight = torch.bmm(concept_query.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim),
                                    embedded.view(bs * seqlen, num_paths,
                                                  self.concept_embed_dim).permute(0, 2, 1))
        c_p_attn_weight = torch.softmax(c_p_attn_weight, dim=2)
        c_p_attn_out = torch.bmm(c_p_attn_weight, embedded.view(bs * seqlen, num_paths, self.concept_embed_dim))
        # c_p_attn_out = c_p_attn_out * c_embed.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim)
        # c_p_attn_out = torch.sum(c_p_attn_out, dim=1) / num.view(-1).unsqueeze(1)
        # c_p_attn_out = torch.mean(c_p_attn_out, dim=1)
        c_p_attn_out = torch.max(c_p_attn_out, dim=1)[0]
        c_p_attn_out = c_p_attn_out.view(bs, seqlen, self.concept_embed_dim)

        return c_p_attn_out


    def forward(self, p_id, c_id, starts, paths, ends, masks, target_c, result, c_embed, cur_result):
        bs = p_id.shape[0]
        seqlen = p_id.shape[1]

        #####################################################################################################
        # models
        # p_id_embed = self.p_id_embed(p_id)

        # code2vec模型
        context_embedded, starts, paths, ends = self.preprocess(paths, starts, ends)
        attn_out = self.attention(c_id, paths, c_embed, context_embedded, bs, seqlen)
        LSTM_input = torch.cat([c_embed, attn_out, cur_result], 2)

        # LSTM
        LSTM_out, final_status = self.LSTM(LSTM_input)
        LSTM_out = LSTM_out.contiguous()

        #######################################################################################################
        # prediction

        # 多知识点，除以知识点个数
        num_concepts = torch.sum(target_c, 2).view(-1)

        prediction = self.predict_Linear(LSTM_out.view(bs * seqlen, -1))
        prediction_1d = torch.bmm(prediction.unsqueeze(1),
                                  target_c.view(bs * seqlen, -1).unsqueeze(2)).squeeze(2)
        mask = num_concepts.gt(0)
        num_concepts = torch.masked_select(num_concepts, mask)
        filtered_pred = torch.masked_select(prediction_1d.squeeze(1), mask)
        filtered_pred = torch.div(filtered_pred, num_concepts)
        filtered_target = torch.masked_select(result.squeeze(1), mask)
        loss = F.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target, attn_out


