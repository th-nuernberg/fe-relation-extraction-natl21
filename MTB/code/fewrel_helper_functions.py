import json

def detokenize_fewrel(text, h, t):
    text_with_ents = []
    for i, token in enumerate(text):
        if i==h[2][0][len(h[2][0])-1]+1:
            text_with_ents.append('[/E1]')
        if i==t[2][0][len(t[2][0])-1]+1:
            text_with_ents.append('[/E2]')
        if i==h[2][0][0]:
            text_with_ents.append('[E1]')
        if i==t[2][0][0]:
            text_with_ents.append('[E2]')
        text_with_ents.append(token)
    return ' '.join(text_with_ents)

def preprocess_fewrel(train_data, test_data, id2rel):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = train_data
    print("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    train_support_set, train_obj_list = process_fewrel_text(text[0])
    data_path = test_data
    
    print("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    test_support_set, test_obj_list = process_fewrel_text(text[0])
    
    data_path = id2rel

    print("Reading id2rel file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()

    id2rel_dict = json.loads(text[0])

    return train_support_set, test_support_set, id2rel_dict, train_obj_list, test_obj_list

def process_fewrel_text(text):
    support_set = dict()
    obj_list = json.loads(text)
    for relation in obj_list.keys():
        support_set[relation] = []
        relation_dict = obj_list[relation]
        for example in relation_dict:
            sent = detokenize_fewrel(example['tokens'], example['h'], example['t'])
            support_set[relation].append(sent)
    return support_set, obj_list