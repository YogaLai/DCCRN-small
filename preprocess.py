import os
import collections
import json


def gen_sorted_json(data_path, json_fn='sort_filename', get_noise=True):
    data_dict = collections.OrderedDict()
    mix = [] 
    clean = []
    noise = []
    for fn in os.listdir(os.path.join(data_path, 'noisy/')):
        id = fn.split('fileid_')[1].split('.')[0]
        data_dict[id]=os.path.join(data_path, 'noisy/'+fn)
    
    data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
    for k, v in data_dict:
        mix.append(v)

    data_dict = collections.OrderedDict()
    for fn in os.listdir(os.path.join(data_path, 'clean')):
        id = fn.split('fileid_')[1].split('.')[0]
        data_dict[id]=os.path.join(data_path, 'clean/'+fn)

    data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
    for k, v in data_dict:
        clean.append(v)

    if get_noise:
        data_dict = collections.OrderedDict()
        for fn in os.listdir(os.path.join(data_path, 'noise')):
            id = fn.split('fileid_')[1].split('.')[0]
            data_dict[id]=os.path.join(data_path, 'noise/'+fn)

        data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
        for k, v in data_dict:
            noise.append(v)
        
        assert len(mix) == len(clean) == len(noise)
        with open(json_fn + '.json', 'w') as f:
            data = {'mix': mix, 'clean': clean, "noise": noise}
            json_dump = json.dumps(data)
            f.write(json_dump)
    else:
        assert len(mix) == len(clean)
        with open(json_fn + '.json', 'w') as f:
            data = {'mix': mix, 'clean': clean}
            json_dump = json.dumps(data)
            f.write(json_dump)

def gen_mix_json(noreverb_path, reverb_path, json_fn):
    data_dict = collections.OrderedDict()
    mix = [] 
    clean = []
    for fn in os.listdir(os.path.join(noreverb_path, 'noisy/')):
        id = fn.split('fileid_')[1].split('.')[0]
        if int(id) < 15000:
            data_dict[id]=os.path.join(noreverb_path, 'noisy/'+fn)
    for fn in os.listdir(os.path.join(reverb_path, 'noisy/')):
        id = fn.split('fileid_')[2].split('.')[0]
        if int(id) >= 15000:
            data_dict[id]=os.path.join(reverb_path, 'noisy/'+fn)
    data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
    for k, v in data_dict:
        mix.append(v)
    
    data_dict = collections.OrderedDict()
    for fn in os.listdir(os.path.join(noreverb_path, 'clean')):
        id = fn.split('fileid_')[1].split('.')[0]
        if int(id) < 15000:
            data_dict[id]=os.path.join(noreverb_path, 'clean/'+fn)
    for fn in os.listdir(os.path.join(reverb_path, 'clean')):
        id = fn.split('fileid_')[1].split('.')[0]
        if int(id) >= 15000:
            data_dict[id]=os.path.join(reverb_path, 'clean/'+fn)
    data_dict = sorted(data_dict.items(), key=lambda x: int(x[0])) # sort by key
    for k, v in data_dict:
        clean.append(v)
    
    assert len(mix) == len(clean)
    with open(json_fn + '.json', 'w') as f:
        data = {'mix': mix, 'clean': clean}
        json_dump = json.dumps(data)
        f.write(json_dump)

def gen_crn_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        mix, clean = data['mix'], data['clean']
    with open('crn_dataset.txt', 'w') as f:
        for i in range(len(mix)):
            f.write(mix[i] + " " + clean[i] + "\n")


if __name__ == '__main__':
    gen_crn_json('sort_filename_30s_250h.json')
    # gen_sorted_json('E:/DNS-Challenge/gen_reverb_dataset_30s_250h', 'sort_filename_30s_250h_with_reverb', get_noise=False)
    # gen_mix_json('E:/DNS-Challenge/gen_testset_30s_250h', 'E:/DNS-Challenge/gen_reverb_dataset_30s_250h', 'sort_filename_reverb_mix_30s_125h_125h')