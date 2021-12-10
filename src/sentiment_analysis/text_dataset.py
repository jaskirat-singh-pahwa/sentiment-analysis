import torch
from torch.utils import data
# from src.sentiment_analysis.constants import Constants


class TextDataset(data.Dataset):
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None):

        assert split in {'train', 'val', 'test'}
        # assert Constants.PAD == "<PAD>"
        # assert Constants.UNK == "<UNK>"
        # assert Constants.END == "<END>"

        self.examples = examples
        self.split = split
        self.threshold = threshold
        self.max_len = max_len
        self.UNK = "<UNK>"
        self.END = "<END>"
        self.PAD = "<PAD>"

        # Dictionaries
        self.idx2word = idx2word
        self.word2idx = word2idx

        if split == 'train':
            self.build_dictionary()

        self.vocab_size = len(self.word2idx)

        # Convert text to indices
        self.textual_ids = []
        self.convert_text()

    def get_frequencies(self):
        word_frequencies = dict()
        for review in self.examples:
            for token in review[1]:
                token = token.lower()
                word_frequencies[token] = word_frequencies.get(token, 0) + 1

        return word_frequencies

    def build_dictionary(self):
        """
        Build the dictionaries idx2word and word2idx. This is only called when split='train', as these
        dictionaries are passed in to the __init__(...) function otherwise. Be sure to use self.threshold
        to control which words are assigned indices in the dictionaries.
        Returns nothing.
        """
        assert self.split == 'train'

        self.idx2word = {0: self.PAD, 1: self.END, 2: self.UNK}
        self.word2idx = {self.PAD: 0, self.END: 1, self.UNK: 2}

        word_frequencies = self.get_frequencies()

        count = 3
        for key, value in word_frequencies.items():
            if value >= self.threshold:
                self.idx2word.update({count: key})
                self.word2idx.update({key: count})
                count += 1

    def convert_text(self):
        """
        Convert each review in the dataset (self.examples) to a list of indices, given by self.word2idx.
        Store this in self.textual_ids; returns nothing.
        """
        word2idx_keys = self.word2idx.keys()

        for review in self.examples:
            review_indices = []
            for token in review[1]:
                if token in word2idx_keys:
                    review_indices.append(self.word2idx[token])
                else:
                    review_indices.append(self.word2idx[self.UNK])

            review_indices.append(self.word2idx[self.END])

            self.textual_ids.append(review_indices)

    def get_text(self, idx):
        """
        Return the review at idx as a long tensor (torch.LongTensor) of integers corresponding to the words
        in the review.
        You may need to pad as necessary (see above).
        """
        review = self.examples[idx][1]
        padded_review = []
        word2idx_keys = self.word2idx.keys()

        if len(review) <= self.max_len:
            missing_indices = [self.word2idx[self.PAD]] * (self.max_len - len(review))
            for token in review:
                if token in word2idx_keys:
                    padded_review.append(self.word2idx[token])
                else:
                    padded_review.append(self.word2idx[self.UNK])

            padded_review.extend(missing_indices)

        else:
            for token in review[0: self.max_len]:
                if token in word2idx_keys:
                    padded_review.append(self.word2idx[token])
                else:
                    padded_review.append(self.word2idx[self.UNK])

        return torch.tensor(padded_review)

    def get_label(self, idx):
        """
        This function should return the value 1 if the label for idx in the dataset is 'positive',
        and 0 if it is 'negative'. The return type should be torch.LongTensor.
        """
        if self.examples[idx][0] == "neg":
            # print(idx, "label = negative")
            return torch.tensor(0)
        elif self.examples[idx][0] == "pos":
            # print(idx, "label = positive")
            return torch.tensor(1)
        else:
            pass

    def __len__(self):
        """
        Return the number of reviews (int value) in the dataset
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Return the review, and label of the review specified by idx.
        """
        return self.get_text(idx), self.get_label(idx)
