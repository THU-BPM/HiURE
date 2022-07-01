import pandas as pd
import json
import re

def generate_relation_dict():
    relation_dict = {}
    id_counter = 0
    for index,item in data.iterrows():
        try:
            k = item[8]
            if k not in relation_dict:
                relation_dict[k] = id_counter
                id_counter+=1
        except:
            print(index,'  ',item)
            pass

    json_str = json.dumps(relation_dict, indent=4)
    with open("nyt_relation2id.json", 'w') as json_file:
        json_file.write(json_str)
    pass

def prepare_entity(data_type,only_relation = False,relation_span = False):
    with open("nyt_relation2id.json", 'r') as json_file:
        relation_dict = json.loads(json_file.read())
    sentence_arr = []
    label_arr = []
    print(data.size)
    print(len(data))
    for index, item in data.iterrows():
        try:
            # find top k [4,0,12,16,5,9,2,6,15,35]
            # find location:4,5,20  business:0,7  people:12,16,2  book:9,6
            if pd.isnull(item[8]) or relation_dict[item[8]] not in [4,3,20,14,7,12,16,2,11,25]:
                continue

            if relation_span:
                ent_type = item[3].split('-')
                new_sentence = item[6].replace(item[1],  '<e1:'+ent_type[0]+'> ' + item[1] + ' </e1:'+ent_type[0]+'>')
                new_sentence = new_sentence.replace(item[2],'<e2:'+ ent_type[1] +'> ' + item[2] + ' </e2:' +ent_type[1]+'>')
            else:
                new_sentence = item[6].replace(item[1], '<e1> ' + item[1] + ' </e1>')
                new_sentence = new_sentence.replace(item[2], '<e2> ' + item[2] + ' </e2>')
            if only_relation:
                ent_type = item[3].split('-')
                new_sentence = new_sentence.replace(item[1],ent_type[0])
                new_sentence = new_sentence.replace(item[2], ent_type[1])

            if not pd.isnull(item[8]):
                sentence_arr.append(new_sentence)
                label_arr.append(relation_dict[item[8]])
            # else:
            #     sentence_arr.append(new_sentence)
            #     label_arr.append(relation_dict['NaN'])
            #     pass
        except Exception as e:
            print(e)
            print(index)
            print(item)
            print()
            pass

    with open(data_type+"_sentence.json", 'w') as json_file:
        json_str = json.dumps(sentence_arr, indent=4)
        json_file.write(json_str)
    with open(data_type+"_label_id.json", 'w') as json_file:
        json_str = json.dumps(label_arr, indent=4)
        json_file.write(json_str)



def cut_data():
    # dp='/home/xuminghu/ACL2021/source_code/NYT-FB_data_process/relation_span_'
    # dp='/home/xuminghu/ACL2021/data/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation-ALL/'
    dp='/home/xuminghu/ACL2021/data/tacred/tacred/'
    suff=['_sentence.json','_label_id.json']
    pref = ['train','test','dev']
    for p in pref:
        for s in suff:
            d = json.load(open(dp+p+s, 'r'))
            with open('/home/xuminghu/ACL2021/cut_data/no_relation_span/tacred/' + p+s, 'w') as json_file:
                json_str = json.dumps(d[:10000], indent=4)
                json_file.write(json_str)
    pass


if __name__ == '__main__':
    cut_data()
    # print('nyt_hello')
    # data_suffix = ['.txt','.test.80%.txt','.validation.20%.txt']
    # data_name = ['train','test','validate']
    # for suffix_index in range(len(data_suffix)):
    #     data_path = '/home/xuminghu/ACL2021/data/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation-ALL'
    #     dataset_path=data_path + '/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation.sortedondate'+data_suffix[suffix_index]
    #     # data_f = open(dataset_path,'r')
    #     # f_lines = data_f.readlines()
    #
    #     data = pd.read_table(
    #         dataset_path,
    #         header=None, encoding='latin1')
    #     # generate_relation_dict()
    #     prepare_entity("related_10_2_"+data_name[suffix_index],relation_span=True)
    pass
