import os
import sys
import json 

import pdb; pdb.trace()

with open(sys.argv[1]) as h:
    for line in h:
        example = json.loads(line.strip())
        if "answerKey" in example:
            label = ord(example["answerKey"]) - ord("A")
            labels.append(label)
        question = example["question"]["stem"]
        assert len(example["question"]["choices"]) == self.args.num_classes
        # format: `<s> Q: Where would I not want a fox? </s> A: hen house </s>`
        question = "Q: " + question
        question_toks = binarize(question, append_bos=True)
        for i, choice in enumerate(example["question"]["choices"]):
            src = "A: " + choice["text"]
            src_bin = torch.cat([question_toks, binarize(src)])
            src_tokens[i].append(src_bin)
            src_lengths[i].append(len(src_bin))
assert all(
    len(src_tokens[0]) == len(src_tokens[i])
    for i in range(self.args.num_classes)
)
assert len(src_tokens[0]) == len(src_lengths[0])
assert len(labels) == 0 or len(labels) == len(src_tokens[0])

for i in range(self.args.num_classes):
    src_lengths[i] = np.array(src_lengths[i])
    src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
    src_lengths[i] = ListDataset(src_lengths[i])

# def load_dataset(
#         self, split, epoch=1, combine=False, data_path=None, return_only=False, **kwargs
#     ):
#         """Load a given dataset split.
#         Args:
#             split (str): name of the split (e.g., train, valid, test)
#         """

#         def binarize(s, append_bos=False):
#             if self.bpe is not None:
#                 s = self.bpe.encode(s)
#             tokens = self.vocab.encode_line(
#                 s,
#                 append_eos=True,
#                 add_if_not_exist=False,
#             ).long()
#             if append_bos and self.args.init_token is not None:
#                 tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
#             return tokens

#         if data_path is None:
#             data_path = os.path.join(self.args.data, split + ".jsonl")
#         if not os.path.exists(data_path):
#             raise FileNotFoundError("Cannot find data: {}".format(data_path))

#         src_tokens = [[] for i in range(self.args.num_classes)]
#         src_lengths = [[] for i in range(self.args.num_classes)]
#         labels = []

#         with open(data_path) as h:
#             for line in h:
#                 example = json.loads(line.strip())
#                 if "answerKey" in example:
#                     label = ord(example["answerKey"]) - ord("A")
#                     labels.append(label)
#                 question = example["question"]["stem"]
#                 assert len(example["question"]["choices"]) == self.args.num_classes
#                 # format: `<s> Q: Where would I not want a fox? </s> A: hen house </s>`
#                 question = "Q: " + question
#                 question_toks = binarize(question, append_bos=True)
#                 for i, choice in enumerate(example["question"]["choices"]):
#                     src = "A: " + choice["text"]
#                     src_bin = torch.cat([question_toks, binarize(src)])
#                     src_tokens[i].append(src_bin)
#                     src_lengths[i].append(len(src_bin))
#         assert all(
#             len(src_tokens[0]) == len(src_tokens[i])
#             for i in range(self.args.num_classes)
#         )
#         assert len(src_tokens[0]) == len(src_lengths[0])
#         assert len(labels) == 0 or len(labels) == len(src_tokens[0])

#         for i in range(self.args.num_classes):
#             src_lengths[i] = np.array(src_lengths[i])
#             src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
#             src_lengths[i] = ListDataset(src_lengths[i])

#         dataset = {
#             "id": IdDataset(),
#             "nsentences": NumSamplesDataset(),
#             "ntokens": NumelDataset(src_tokens[0], reduce=True),
#         }

#         for i in range(self.args.num_classes):
#             dataset.update(
#                 {
#                     "net_input{}".format(i + 1): {
#                         "src_tokens": RightPadDataset(
#                             src_tokens[i],
#                             pad_idx=self.source_dictionary.pad(),
#                         ),
#                         "src_lengths": src_lengths[i],
#                     }
#                 }
#             )