import torch

class Vocab():
    def __init__(self, embed, word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'

    def __len__(self):
        return len(word2id)

    def i2w(self, idx):
        return self.id2word[idx]

    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def make_features(self, batch, device, sent_trunc=2002, doc_trunc=100, split_token='</s> '):
        input, sents_list, targets, doc_lens = [], [], [], []
        # trunc document

        #for input, doc, label in zip(batch['input'], batch['doc'], batch['labels']):
            #sents = [doc.split(split_token)
            #labels = label.split(split_token)
            #labels = [int(l) for l in labels]
            #max_sent_num = min(doc_trunc, len(sents))
            #sents = sents[:max_sent_num]
            #labels = labels[:max_sent_num]
            #sents_list += sents
            #targets += labels
            #doc_lens.append(len(sents))

        """
        features(다섯 줄) 
        """
        # trunc or pad sent
        batch_sents = []
        # trunc or pad sent
        for input in batch['doc']:
            words = input.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            batch_sents.append(words)

        doc_features = []
        for input in batch_sents:
            feature = [self.SOS_IDX]+ [self.w2i(w) for w in input][:sent_trunc-2] + [self.EOS_IDX]
            feature += [self.PAD_IDX for _ in range(sent_trunc - len(feature))]
            doc_features.append(feature)


        """
        input(한줄)
        """
        batch_inputs = []
        max_sent_len = 0
        # trunc or pad sent
        for input in batch['input']:
            max_sent_len = 0
            words = input.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_inputs.append(words)

        input_features = []
        for input in batch_inputs:
            feature = [self.SOS_IDX]+ [self.w2i(w) for w in input][:sent_trunc-2] + [self.EOS_IDX]
            feature += [self.PAD_IDX for _ in range(sent_trunc - len(feature))]
            input_features.append(feature)


        input = torch.LongTensor(input_features)
        features = torch.LongTensor(doc_features)
        #targets = torch.LongTensor(targets)

        return input, features #, targets, doc_lens

    def make_predict_features(self, batch, sent_trunc=150, doc_trunc=100, split_token='. '):
        sents_list, doc_lens = [], []
        for doc in batch:
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)

        return features, doc_lens