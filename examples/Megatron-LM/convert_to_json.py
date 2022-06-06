
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



