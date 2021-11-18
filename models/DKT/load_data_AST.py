import numpy as np
from tqdm import tqdm
import math
import json

class DATA_AST():
    def __init__(self, num_concepts, seqlen, separate_char=';'):
        super(DATA_AST, self).__init__()

        self.n_concept = num_concepts
        self.seqlen = seqlen
        self.separate_char = separate_char

    def load_data(self, path):
        f_data = open(path, 'r')
        seq_len_data = []
        p_id_data = []
        c_id_data = []
        source_token_data = []
        path_token_data = []
        target_token_data = []
        context_valid_masks_data = []
        result_data = []
        target_c_data = []
        x_result_data = []

        for lineID, line in enumerate(tqdm(f_data)):

            # if lineID > 23:
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

            elif lineID % 8 == 4:
                temp = line.split(self.separate_char)
                source_tokens = []
                path_tokens = []
                target_tokens = []
                context_valid_masks = []
                for i in range(len(temp)):
                    cur_source_tokens = []
                    cur_path_tokens = []
                    cur_target_tokens = []
                    cur_context_masks = []
                    temp[i] = json.loads(temp[i])
                    for path in temp[i]:
                        try:
                            source_token, path_token, target_token = map(int, path.strip().split(', '))
                        except:
                            continue

                        cur_source_tokens.append(source_token)
                        cur_path_tokens.append(path_token)
                        cur_target_tokens.append(target_token)
                        cur_context_masks.append(1)

                    source_tokens.append(cur_source_tokens)
                    path_tokens.append(cur_path_tokens)
                    target_tokens.append(cur_target_tokens)
                    context_valid_masks.append(cur_context_masks)

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
                    source_token_sequence = []
                    path_token_sequence = []
                    target_token_sequence = []
                    context_valid_masks_sequence = []
                    result_sequence = []

                    if k == n_split - 1:
                        end_index = len(problem_id)
                    else:
                        end_index = (k + 1) * new_seq_len
                    for i in range(k * new_seq_len, end_index):
                        if problem_id[i] > 0:
                            p_id_sequence.append(int(problem_id[i]))
                            c_id_sequence.append(c_tags[i])
                            source_token_sequence.append(source_tokens[i])
                            path_token_sequence.append(path_tokens[i])
                            target_token_sequence.append(target_tokens[i])
                            context_valid_masks_sequence.append(context_valid_masks[i])
                            result_sequence.append(result[i])

                    if len(p_id_sequence) > 2:
                        p_id_data.append(p_id_sequence[:-1])
                        c_id_data.append(c_id_sequence[:-1])
                        source_token_data.append(source_token_sequence[:-1])
                        path_token_data.append(path_token_sequence[:-1])
                        target_token_data.append(target_token_sequence[:-1])
                        context_valid_masks_data.append(context_valid_masks_sequence[:-1])
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

        source_token_dataArray = np.zeros((len(source_token_data), self.seqlen, 200))
        for i in range(len(source_token_data)):
            for j in range(len(source_token_data[i])):
                dat = source_token_data[i][j]
                source_token_dataArray[i, j, :len(dat)] = dat

        path_token_dataArray = np.zeros((len(path_token_data), self.seqlen, 200))
        for i in range(len(path_token_data)):
            for j in range(len(path_token_data[i])):
                dat = path_token_data[i][j]
                path_token_dataArray[i, j, :len(dat)] = dat

        target_token_dataArray = np.zeros((len(target_token_data), self.seqlen, 200))
        for i in range(len(target_token_data)):
            for j in range(len(target_token_data[i])):
                dat = target_token_data[i][j]
                target_token_dataArray[i, j, :len(dat)] = dat

        context_valid_mask_dataArray = np.zeros((len(context_valid_masks_data), self.seqlen, 200))
        for i in range(len(context_valid_masks_data)):
            for j in range(len(context_valid_masks_data[i])):
                dat = context_valid_masks_data[i][j]
                context_valid_mask_dataArray[i, j, :len(dat)] = dat


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


        return p_id_dataArray, c_id_dataArray, source_token_dataArray, path_token_dataArray, \
               target_token_dataArray, context_valid_mask_dataArray, \
               target_c_dataArray, result_dataArray, c_embed_dataArray, x_result_dataArray