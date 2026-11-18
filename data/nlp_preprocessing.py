import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        vocab = set()
        n_words = 0
        for pos, neg in zip(positive, negative):
            pos_words, neg_words = pos.split(), neg.split()
            n_words = max(n_words, len(pos_words), len(neg_words))
            vocab = vocab | set(pos_words) | set(neg_words)

        sorted_vocab = sorted(list(vocab))
        print(sorted_vocab)
        word_idx = {sorted_vocab[i]: i+1 for i in range(len(sorted_vocab))}
        print(word_idx)

        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        final_tensors = []
        for s in positive:
            temp = []
            for word in s.split():
                temp.append(word_idx[word])
            final_tensors.append(torch.tensor(temp))
        
        for s in negative:
            temp = []
            for word in s.split():
                temp.append(word_idx[word])
            final_tensors.append(torch.tensor(temp))

        # 4. Pad shorter sequences with 0s using 
        return nn.utils.rnn.pad_sequence(final_tensors, batch_first=True)
        
