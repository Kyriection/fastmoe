import json 
import sys 
import jsonlines

json_data = []
f = open(sys.argv[1], 'r', encoding='utf-8')
alllines = f.readlines()

for line in alllines:
    if len(line) == 1: continue
    if line[-1] == '\n':
        json_data.append({'text': line[:-1]})

with jsonlines.open('bookcorpus.json', 'w') as writer:
    writer.write_all(json_data)



# print(alllines[0]=='\n')
# print(len(alllines[6]))



# print(alllines[0:10])

# data = json.load(f)


# with open()


# print(data)


# import os
# import sys
# from glob import glob

# from blingfire import text_to_sentences

# file_dir = sys.argv[1]


# def convert_into_sentences(lines):
#     stack = []
#     sent_L = []
#     n_sent = 0
#     for chunk in lines:
#         if not chunk.strip():
#             if stack:
#                 sents = text_to_sentences(
#                     " ".join(stack).strip().replace('\n', ' ')).split('\n')
#                 sent_L.extend(sents)
#                 n_sent += len(sents)
#                 sent_L.append('\n')
#                 stack = []
#             continue
#         stack.append(chunk.strip())

#     if stack:
#         sents = text_to_sentences(
#             " ".join(stack).strip().replace('\n', ' ')).split('\n')
#         sent_L.extend(sents)
#         n_sent += len(sents)
#     return sent_L, n_sent


# file_list = list(sorted(glob(os.path.join(file_dir, '*.txt'))))

# for i, file_path in enumerate(file_list):
#     sents, n_sent = convert_into_sentences(open(file_path).readlines())
#     print('\n'.join(sents))
#     print('\n\n\n\n')
#     sys.stderr.write(
#         '{}/{}\t{}\t{}\n'.format(i, len(file_list), n_sent, file_path))