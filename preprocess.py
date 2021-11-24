import os
import collections
import json

def gen_sorted_json(data_path, json_fn='sort_filename'):
    data_dict = collections.OrderedDict()
    noisy = [] 
    clean = []
    noise = []
    for fn in os.listdir(os.path.join(data_path, 'noisy')):
        id = fn.split('fileid_')[1].split('.')[0]
        data_dict[id]=fn
    
    data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
    for k, v in data_dict:
        noisy.append(v)

    data_dict = collections.OrderedDict()
    for fn in os.listdir(os.path.join(data_path, 'clean')):
        id = fn.split('fileid_')[1].split('.')[0]
        data_dict[id]=fn

    data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
    for k, v in data_dict:
        clean.append(v)

    data_dict = collections.OrderedDict()
    for fn in os.listdir(os.path.join(data_path, 'noise')):
        id = fn.split('fileid_')[1].split('.')[0]
        data_dict[id]=fn

    data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
    for k, v in data_dict:
        noise.append(v)
    
    with open(json_fn + '.json', 'w') as f:
        data = {'noisy': noisy, 'clean': clean, "noise": noise}
        json_dump = json.dumps(data)
        f.write(json_dump)

if __name__ == '__main__':
    gen_sorted_json('E:/DNS-Challenge/gen_dataset')