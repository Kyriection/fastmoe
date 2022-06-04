import json 
import sys 

f = open(sys.argv[1], 'r', encoding='utf-8')

data = json.load(f)


print(data)