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
        self.LSTM_feature_dim = 2 + num_concepts + 1 + node_embed_dim

        self.LSTM = nn.LSTM(input_size=self.LSTM_feature_dim, hidden_size=hidden_dim,
                            num_layers=hidden_layers, batch_first=True)
        self.predict_Linear = nn.Linear(hidden_dim, num_concepts + 1, bias=True)


    def init_embeddings(self):
        nn.init.kaiming_normal_(self.p_id_embed.weight)


    def init_params(self):
        nn.init.kaiming_normal_(self.predict_Linear.weight)
        nn.init.constant_(self.predict_Linear.bias, 0)


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

        codevec = []
        for i in range(bs):
            for j in range(seqlen):
                out = self.ggnnlayer(node_embed[i, j], edge[i, j], edge_weight[i, j])
                gate = variable(torch.zeros(out.size(0), dtype=torch.long), self.gpu)
                code_embedding = self.pool(out, batch=gate).squeeze(0)
                codevec.append(code_embedding)

        codevec = torch.cat([codevec[i].unsqueeze(0) for i in range(bs * seqlen)], 0).view(bs, seqlen, -1)

        # PKT
        # p_id_embed = self.p_id_embed(p_id)
        LSTM_input = torch.cat([c_embed, codevec, cur_result], 2)

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


