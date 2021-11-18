import numpy as np
from tqdm import tqdm
import math
import json

class DATA_ggnn():
    def __init__(self, num_concepts, seqlen, separate_char=';'):
        super(DATA_ggnn, self).__init__()

        self.n_concept = num_concepts
        self.seqlen = seqlen
        self.separate_char = separate_char


    def load_data(self, path):
        f_data = open(path, 'r')
        seq_len_data = []
        p_id_data = []
        c_id_data = []
        node_id_data = []
        edge_data = []
        edge_type_data = []
        result_data = []
        target_c_data = []
        x_result_data = []

        for lineID, line in enumerate(tqdm(f_data)):

            # if lineID > 63:
            #     continue

            line = line.strip()
            if lineID % 8 == 0:
                line = line.split(',')
                seq_len_data.append(int(line[1]))

            elif lineID % 8 == 1:
                problem_id = []
                P = line.split(self.separate_char)
                if len(P[len(P) - 1]) == 0:
                    P = P[:-1]
                for i in range(len(P)):
                    problem_id.append(int(P[i]))

            elif lineID % 8 == 2:
                c_tags = []
                C = line.split(self.separate_char)
                if len(C[len(C) - 1]) == 0:
                    C = C[:-1]
                for i in range(len(C)):
                    tag = json.loads(C[i])
                    tag = [int(m) for m in tag]
                    c_tags.append(tag)

            elif lineID % 8 == 3:
                node_ids = []
                line = line.split(self.separate_char)
                for i in range(len(line)):
                    node_id = json.loads(line[i])
                    node_ids.append(node_id)

            elif lineID % 8 == 4:
                edges = []
                line = line.split(self.separate_char)
                for i in range(len(line)):
                    edge = json.loads(line[i])
                    edges.append(edge)

            elif lineID % 8 == 5:
                edge_types = []
                line = line.split(self.separate_char)
                for i in range(len(line)):
                    edge_type = json.loads(line[i])
                    edge_types.append(edge_type)

            elif lineID % 8 == 6:
                result = []
                res = line.split(self.separate_char)
                if len(res[len(res) - 1]) == 0:
                    res = res[:-1]
                for i in range(len(res)):
                    result.append(int(res[i]))

                new_seq_len = self.seqlen + 1
                n_split = 1
                if len(problem_id) > new_seq_len:
                    n_split = math.floor(len(problem_id) / new_seq_len)
                    if len(problem_id) % new_seq_len:
                        n_split = n_split + 1


                for k in range(n_split):
                    p_id_sequence = []
                    c_id_sequence = []
                    node_id_dequence = []
                    edge_sequence = []
                    edge_type_sequence = []
                    result_sequence = []

                    if k == n_split - 1:
                        end_index = len(problem_id)
                    else:
                        end_index = (k + 1) * new_seq_len
                    for i in range(k * new_seq_len, end_index):
                        if problem_id[i] > 0:
                            p_id_sequence.append(int(problem_id[i]))
                            c_id_sequence.append(c_tags[i])
                            node_id_dequence.append(node_ids[i])
                            edge_sequence.append(edges[i])
                            edge_type_sequence.append(edge_types[i])
                            result_sequence.append(result[i])

                    if len(p_id_sequence) > 2:
                        p_id_data.append(p_id_sequence[:-1])
                        c_id_data.append(c_id_sequence[:-1])
                        node_id_data.append(node_id_dequence[:-1])
                        edge_data.append(edge_sequence[:-1])
                        edge_type_data.append(edge_type_sequence[:-1])
                        result_data.append(result_sequence[1:])
                        target_c_data.append(c_id_sequence[1:])
                        x_result_data.append(result_sequence[:-1])

        f_data.close()

        p_id_dataArray = np.zeros((len(p_id_data), self.seqlen))
        for i in range(len(p_id_data)):
            dat = p_id_data[i]
            p_id_dataArray[i, :len(dat)] = dat

        # onehot形式
        c_id_dataArray = np.zeros((len(c_id_data), self.seqlen, 2 * self.n_concept + 1))
        for i in range(len(c_id_data)):
            for j in range(len(c_id_data[i])):
                for k in range(len(c_id_data[i][j])):
                    c_id_dataArray[i, j, c_id_data[i][j][k] + x_result_data[i][j] * self.n_concept] = 1

        c_embed_dataArray = np.zeros((len(c_id_data), self.seqlen, self.n_concept + 1))
        for i in range(len(c_id_data)):
            for j in range(len(c_id_data[i])):
                for k in range(len(c_id_data[i][j])):
                    c_embed_dataArray[i, j, c_id_data[i][j][k]] = 1

        node_id_dataArray = np.zeros((len(node_id_data), self.seqlen, 200))
        for i in range(len(node_id_data)):
            dat = node_id_data[i]
            for j in range(len(dat)):
                node_id_dataArray[i, j, :min(len(dat[j]), 200)] = dat[j][:min(len(dat[j]), 200)]

        edge_dataArray = np.zeros((len(edge_data), self.seqlen, 2, 200))
        for i in range(len(edge_data)):
            for j in range(len(edge_data[i])):
                dat = edge_data[i][j]
                for k in range(2):
                    edge_dataArray[i, j, k, :min(len(dat[k]), 200)] = dat[k][:min(len(dat[k]), 200)]

        edge_type_dataArray = np.zeros((len(edge_type_data), self.seqlen, 200))
        for i in range(len(edge_type_data)):
            dat = edge_type_data[i]
            for j in range(len(dat)):
                edge_type_dataArray[i, j, :min(len(dat[j]), 200)] = dat[j][:min(len(dat[j]), 200)]

        # target and label
        target_c_dataArray = np.zeros((len(target_c_data), self.seqlen, self.n_concept + 1))
        for i in range(len(target_c_data)):
            for j in range(len(target_c_data[i])):
                target_c_dataArray[i, j, target_c_data[i][j]] = 1

        result_dataArray = np.zeros((len(result_data), self.seqlen))
        for i in range(len(result_data)):
            dat = result_data[i]
            result_dataArray[i, :len(dat)] = dat

        x_result_dataArray = np.zeros((len(x_result_data), self.seqlen, 2))
        for i in range(len(x_result_data)):
            for j in range(len(x_result_data[i])):
                x_result_dataArray[i, j, x_result_data[i][j]] = 1


        return p_id_dataArray, c_id_dataArray, node_id_dataArray, edge_dataArray, edge_type_dataArray, \
               target_c_dataArray, result_dataArray, c_embed_dataArray, x_result_dataArray
