import enum
from torch.utils.data import  Dataset
from torch.utils.data._utils.collate import default_collate
from typing import Dict, List, Tuple, Union
import json
from transformers import PreTrainedTokenizer
import numpy as np
import collections
from tqdm import tqdm
import re
import copy


FCInst = collections.namedtuple('FCInst', 'fid question answer facts entity')
FCFeat = collections.namedtuple('FCFeat', 'fid input_ids question_ids flag is_training')

"""def convert_instances_to_feature_tensors(instances: List[FCInst],
                                         tokenizer: PreTrainedTokenizer,
                                         max_seq_length: int,
                                         is_training: bool=False,
                                         sep_token_extra: bool = False,
                                         max_answer_length: int=20, model_num=None, task=None, tokenizer1=None) -> List[FCFeat]:
    features = []
    for inst in tqdm(instances):
        ## question tokenization
        question = inst.question
        token_question = tokenizer.tokenize(question)
        idx_to_question_token_id = []
        temp_question = question
        for token_id, token in enumerate(token_question):
            if token.startswith("##"):
                token = token[2:]
            while re.match(r'\s', temp_question):
                temp_question = temp_question[1:]
                idx_to_question_token_id.append(-1)
            for _ in range(len(token)):
                idx_to_question_token_id.append(token_id)
            temp_question = temp_question[len(token):]
        for _ in range(len(temp_question)):
            idx_to_question_token_id.append(-1)

        if len(idx_to_question_token_id) != len(question):
            continue

        ## fact tokenization
        f = False
        token_facts = []
        element_fact_positions = []
        element_question_positions = []
        for fact, [_, entities] in zip(inst.facts, inst.entity):
            token_fact = tokenizer.tokenize(fact)
            idx_to_fact_token_id = []
            temp_fact = fact
            for token_id, token in enumerate(token_fact):
                if token.startswith("##"):
                    token = token[2:]
                while re.match(r'\s', temp_fact):
                    temp_fact = temp_fact[1:]
                    idx_to_fact_token_id.append(-1)
                for _ in range(len(token)):
                    idx_to_fact_token_id.append(token_id)
                temp_fact = temp_fact[len(token):]
            for _ in range(len(temp_fact)):
                idx_to_fact_token_id.append(-1)

            if len(idx_to_fact_token_id) != len(fact):
                f = True
                break

            for entity in entities:
                f_element, f_startOffset = entity[0]
                q_element, q_startOffset = entity[1]

                f_endOffset = f_startOffset + len(f_element) - 1
                f_start_position = idx_to_fact_token_id[f_startOffset] + len(token_facts)
                f_end_position = idx_to_fact_token_id[f_endOffset] + len(token_facts)
                element_fact_positions.append((f_start_position, f_end_position))

                q_endOffset = q_startOffset + len(q_element) - 1
                q_start_position = idx_to_question_token_id[q_startOffset]
                q_end_position = idx_to_question_token_id[q_endOffset]
                element_question_positions.append((q_start_position, q_end_position))

            token_facts += token_fact

        if f:
            continue

        input_sequence = " ".join(inst.facts) + " [SEP] " + inst.answer
        input_tokens = token_facts + tokenizer.tokenize("[SEP]"+inst.answer)
        flag = [[0]*(len(input_tokens)+2)]

        for f_positions in element_fact_positions:
            f_start, f_end = f_positions
            for f_idx in range(f_start+1, f_end+2):
                flag[0][f_idx] = 1

        if is_training:
            for _ in range(len(token_question)-1):
                temp = copy.deepcopy(flag[0])
                flag.append(temp)
            for f_positions, q_positions in zip(element_fact_positions, element_question_positions):
                f_start, f_end = f_positions
                q_start, _ = q_positions
                for q_idx in range(len(flag)):
                    if q_idx > q_start:
                        for f_idx in range(f_start+1, f_end+2):
                            flag[q_idx][f_idx] = 2

        batch_input = tokenizer.encode(input_sequence, add_special_tokens=True, truncation=True)
        target_ids = tokenizer.encode(question, add_special_tokens=True, truncation=True)

        assert len(batch_input) == len(flag[0]), f"{len(batch_input)} | {len(flag[0])}"
        if is_training:
            assert len(target_ids)-2 == len(flag), f"{len(target_ids)-2} | {len(flag)}"

        features.append(FCFeat(fid=inst.fid,
                        input_ids=batch_input,
                        question_ids=target_ids,
                        flag = flag,
                        is_training=is_training))
    return features """

def convert_instances_to_feature_tensors(instances: List[FCInst],
                                         tokenizer: PreTrainedTokenizer,
                                         max_seq_length: int,
                                         is_training: bool=False,
                                         sep_token_extra: bool = False,
                                         max_answer_length: int=20, model_num=None, task=None, tokenizer1=None) -> List[FCFeat]:
    features = []
    for inst in tqdm(instances):
        ## question tokenization
        question = inst.question
        token_question = tokenizer.tokenize(question)
        idx_to_question_token_id = []
        temp_question = question
        for token_id, token in enumerate(token_question):
            if token.startswith("##"):
                token = token[2:]
            while re.match(r'\s', temp_question):
                temp_question = temp_question[1:]
                idx_to_question_token_id.append(-1)
            for _ in range(len(token)):
                idx_to_question_token_id.append(token_id)
            temp_question = temp_question[len(token):]
        for _ in range(len(temp_question)):
            idx_to_question_token_id.append(-1)

        if len(idx_to_question_token_id) != len(question):
            continue

        ## fact tokenization
        f = False
        token_facts = []
        element_fact_positions = []
        element_question_positions = []
        for fact, [_, entities] in zip(inst.facts, inst.entity):
            token_fact = tokenizer.tokenize(fact)
            idx_to_fact_token_id = []
            temp_fact = fact
            for token_id, token in enumerate(token_fact):
                if token.startswith("##"):
                    token = token[2:]
                while re.match(r'\s', temp_fact):
                    temp_fact = temp_fact[1:]
                    idx_to_fact_token_id.append(-1)
                for _ in range(len(token)):
                    idx_to_fact_token_id.append(token_id)
                temp_fact = temp_fact[len(token):]
            for _ in range(len(temp_fact)):
                idx_to_fact_token_id.append(-1)

            if len(idx_to_fact_token_id) != len(fact):
                f = True
                break

            for entity in entities:
                f_element, f_startOffset = entity[0]
                q_element, q_startOffset = entity[1]

                f_endOffset = f_startOffset + len(f_element) - 1
                f_start_position = idx_to_fact_token_id[f_startOffset] + len(token_facts)
                f_end_position = idx_to_fact_token_id[f_endOffset] + len(token_facts)
                element_fact_positions.append((f_start_position, f_end_position))

                q_endOffset = q_startOffset + len(q_element) - 1
                q_start_position = idx_to_question_token_id[q_startOffset]
                q_end_position = idx_to_question_token_id[q_endOffset]
                element_question_positions.append((q_start_position, q_end_position))

            token_facts += token_fact

        if f:
            continue

        input_sequence = " ".join(inst.facts) + " [SEP] " + inst.answer
        input_tokens = token_facts + tokenizer.tokenize("[SEP]"+inst.answer)
        flag = [[0]*(len(input_tokens)+2)]

        for (s, e) in element_fact_positions:
            for idx in range(s, e+1):
                flag[0][idx+1] = 1

        for i, _ in enumerate(token_question):
            if i == 0:
                continue
            temp = copy.deepcopy(flag[i-1])
            for (f_s, f_e), (q_s, q_e) in zip(element_fact_positions, element_question_positions):
                if i-1 >= q_s and i-1 <= q_e:
                    for idx in range(f_s, f_e+1):
                        if token_facts[idx] == token_question[i-1] and temp[idx+1] == 1:
                            temp[idx+1] = 2
            flag.append(temp)

        batch_input = tokenizer.encode(input_sequence, add_special_tokens=True, truncation=True)
        target_ids = tokenizer.encode(question, add_special_tokens=True, truncation=True)

        decoded = tokenizer.decode(batch_input)

        assert len(batch_input) == len(flag[0]), f"{len(batch_input)} | {len(flag[0])}"
        if is_training:
            assert len(target_ids)-2 == len(flag), f"{len(target_ids)-2} | {len(flag)}"

        features.append(FCFeat(fid=inst.fid,
                        input_ids=batch_input,
                        question_ids=target_ids,
                        flag = flag,
                        is_training=is_training))
    return features 

class FCDataset(Dataset):

    def __init__(self, file: Union[str, None], tokenizer: PreTrainedTokenizer,
                 pretrain_model_name: str,
                 number: int = -1,
                 max_question_len: int = 100,
                 max_answer_length: int = 30,
                 model_num=None,
                 tokenizer1=None,
                 mode='generation',
                 is_training=False) -> None:
        insts = []
        self.fid_to_qid = dict()
        self.skip_num = 0
        self.tokenizer = tokenizer
        self.pretrain_model_name = pretrain_model_name
        print(f"[Data Info] Reading file: {file}")
        #print(file)
        with open(file, 'r', encoding='utf-8') as read_file:
            data = json.load(read_file)
        if number >= 0:
            data = data[:number]

        for fid, sample in tqdm(enumerate(data)):
            #a_entities = []
            #for _, entities in sample["a_entities"]:
            #    a_entities += entities
            #if len(a_entities) == 0:
            #    self.skip_num += 1
            #    continue

            qid = sample["_id"]
            self.fid_to_qid[fid] = qid
            insts.append(FCInst(fid=fid,
                                question=str(sample['question']),
                                answer=str(sample['answer']),
                                facts = sample['supporting_facts'],
                                entity = sample['q_entities']))
            
        self._features = convert_instances_to_feature_tensors(instances=insts,
                                                              tokenizer=tokenizer,
                                                              max_seq_length=max_question_len,
                                                              sep_token_extra= "roberta" in pretrain_model_name or "bart" in pretrain_model_name or "checkpoint" in pretrain_model_name,
                                                              max_answer_length=max_answer_length, model_num=model_num, is_training=is_training)
        
        print("Skipped Samples:", self.skip_num)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> FCFeat:
        return self._features[idx]

    def collate_fn(self, batch: List[FCFeat]):
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        max_question_length = max([len(feature.question_ids) for feature in batch])
        #print(batch[0].entity, batch[0].label)
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            pad_label_length = max_question_length - len(feature.question_ids)
            flag = feature.flag
            flag = [item + [0] * (max_wordpiece_length - len(item)) for item in flag]
            if feature.is_training:
                flag = flag + [[0] * max_wordpiece_length] * (max_question_length - len(flag))
            #pad_label_length = max_label_length - len(feature.label)
            batch[i] = FCFeat(fid=feature.fid,
                              input_ids=np.asarray(feature.input_ids + [0] * padding_length),
                              question_ids=np.asarray(feature.question_ids + [0] * pad_label_length),
                              flag=np.asarray(flag),
                              is_training=np.asarray(feature.is_training))
        #print(batch)
        results = FCFeat(*(default_collate(samples) for samples in zip(*batch)))
        return results