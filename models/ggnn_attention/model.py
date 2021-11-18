import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GatedGraphConv
from torch_geometric.nn.glob import GlobalAttention

from utils import *
import random

class MODEL(nn.Module):
    def __init__(self, num_concepts, num_problems, hidden_dim, hidden_layers,
                 concept_embed_dim, vocablen, node_embed_dim, num_nodes, gpu):
        super(MODEL, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.gpu = gpu
        self.num_concepts = num_concepts
        self.node_embed_dim = node_embed_dim
        self.concept_embed_dim = concept_embed_dim
        self.num_nodes = num_nodes

        # ggnn network
        self.node_embed = nn.Embedding(vocablen, node_embed_dim)
        self.edge_embed = nn.Embedding(8, node_embed_dim, padding_idx=0)
        self.ggnnlayer = GatedGraphConv(node_embed_dim, num_layers=4)
        self.mlp_gate = nn.Sequential(nn.Linear(node_embed_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)

        # DKT network
        self.p_id_embed = nn.Embedding(num_problems + 1, concept_embed_dim)

        # input feature dim
        self.LSTM_feature_dim = 2 + num_concepts + 1 + concept_embed_dim

        self.LSTM = nn.LSTM(input_size=self.LSTM_feature_dim, hidden_size=hidden_dim,
                            num_layers=hidden_layers, batch_first=True)
        self.predict_Linear = nn.Linear(hidden_dim, num_concepts + 1, bias=True)

        self.concept_embedding = nn.Parameter(torch.empty(num_concepts + 1, concept_embed_dim), requires_grad=True)
        self.transdim = nn.Linear(node_embed_dim, concept_embed_dim, bias=False)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.p_id_embed.weight)
        nn.init.kaiming_normal_(self.concept_embedding.data)


    def init_params(self):
        nn.init.kaiming_normal_(self.predict_Linear.weight)
        nn.init.constant_(self.predict_Linear.bias, 0)
        nn.init.kaiming_normal_(self.transdim.weight)

    def attention(self, c_id, c_embed, ggnn, bs, seqlen):
        if(ggnn.shape[3] != self.concept_embed_dim):
            ggnn = self.transdim(ggnn)

        num_nodes = ggnn.shape[2]

        c_embed = c_embed.unsqueeze(3).expand(bs, seqlen, self.num_concepts + 1, self.concept_embed_dim)

        # the query is concept
        # 习题知识点个数，用于c_p_attn的AvgPooling
        num = torch.sum(c_id, dim=2).unsqueeze(2)
        num = torch.masked_fill(num, num.eq(0), 1)
        concept_query = c_embed * self.concept_embedding.repeat(bs, seqlen, 1, 1)
        c_p_attn_weight = torch.bmm(concept_query.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim),
                                    ggnn.view(bs * seqlen, num_nodes,
                                                   self.concept_embed_dim).permute(0, 2, 1))
        c_p_attn_weight = torch.softmax(c_p_attn_weight, dim=2)
        c_p_attn_out = torch.bmm(c_p_attn_weight, ggnn.view(bs * seqlen, num_nodes, self.concept_embed_dim))
        c_p_attn_out = c_p_attn_out * c_embed.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim)
        c_p_attn_out = torch.sum(c_p_attn_out, dim=1) / num.view(-1).unsqueeze(1)
        c_p_attn_out = c_p_attn_out.view(bs, seqlen, self.concept_embed_dim)

        return c_p_attn_out


    def attention_all(self, c_id, c_embed, ggnn, bs, seqlen):
        if(ggnn.shape[3] != self.concept_embed_dim):
            ggnn = self.transdim(ggnn)

        num_nodes = ggnn.shape[2]

        # c_embed = c_embed.unsqueeze(3).expand(bs, seqlen, self.num_concepts + 1, self.concept_embed_dim)
        c_embed = variable(torch.ones(bs, seqlen, self.num_concepts + 1, self.concept_embed_dim), self.gpu)

        # the query is concept
        # 习题知识点个数，用于c_p_attn的AvgPooling
        # num = torch.sum(c_id, dim=2).unsqueeze(2)
        # num = torch.masked_fill(num, num.eq(0), 1)
        concept_query = c_embed * self.concept_embedding.repeat(bs, seqlen, 1, 1)
        c_p_attn_weight = torch.bmm(concept_query.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim),
                                    ggnn.view(bs * seqlen, num_nodes,
                                                   self.concept_embed_dim).permute(0, 2, 1))
        c_p_attn_weight = torch.softmax(c_p_attn_weight, dim=2)
        c_p_attn_out = torch.bmm(c_p_attn_weight, ggnn.view(bs * seqlen, num_nodes, self.concept_embed_dim))
        # c_p_attn_out = c_p_attn_out * c_embed.view(bs * seqlen, self.num_concepts + 1, self.concept_embed_dim)
        # c_p_attn_out = torch.sum(c_p_attn_out, dim=1) / num.view(-1).unsqueeze(1)
        # c_p_attn_out = torch.mean(c_p_attn_out, dim=1)
        c_p_attn_out = torch.max(c_p_attn_out, dim=1)[0]
        c_p_attn_out = c_p_attn_out.view(bs, seqlen, self.concept_embed_dim)

        return c_p_attn_out


    def forward(self, p_id, c_id, node_id, edge, edge_type, target_c, result, c_embed, cur_result):
        bs = p_id.shape[0]
        seqlen = p_id.shape[1]
        nodelen = node_id.shape[2]

        if self.num_nodes < 200:
            sample = random.sample(range(0, node_id.shape[2]), self.num_nodes)
            edge = edge[:, :, :, sample]
            edge_type = edge_type[:, :, sample]

        #####################################################################################################
        # ggnn
        node_embed = self.node_embed(node_id).view(bs, seqlen, nodelen, -1)
        edge_weight = self.edge_embed(edge_type).mean(3).unsqueeze(3)

        batch_ggnn = []
        for i in range(bs):
            ggnn = []
            for j in range(seqlen):
                out = self.ggnnlayer(node_embed[i, j], edge[i, j], edge_weight[i, j])
                ggnn.append(out)
            ggnn = torch.cat([ggnn[i].unsqueeze(0) for i in range(seqlen)], 0)
            batch_ggnn.append(ggnn)

        batch_ggnn = torch.cat([batch_ggnn[i].unsqueeze(0) for i in range(bs)], 0)

        attn_out = self.attention(c_id, c_embed, batch_ggnn, bs, seqlen)

        # PKT
        # p_id_embed = self.p_id_embed(p_id)
        LSTM_input = torch.cat([c_embed, attn_out, cur_result], 2)

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


