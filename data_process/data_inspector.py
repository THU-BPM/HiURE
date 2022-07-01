import json
import pandas as pd

def inspect_category():
    # tacred: [0, 2443, 1890, 468, 152, 390, 445, 211, 808, 72, 374, 325, 124, 1524, 165, 104, 111, 53, 49, 286, 296, 179, 331, 170, 117, 122, 75, 28, 76, 229, 81, 63, 258, 382, 134, 149, 105, 6, 91, 38, 65, 23]
    # NYT: 1~10000+
    f_name = 'test_label_with_0_train_label_id.json'
    # f_name = '/home/xuminghu/ACL2021/data/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation-ALL/train_label_id.json'
    # location:4,3,20  business:14,7  people:12,16,2  book:11,25
    with open(f_name, 'r') as json_file:
        relation_dict = json.loads(open("nyt_relation2id.json", 'r').read())
        relation_arr = ['']*len(relation_dict)
        for rel in relation_dict:
            relation_arr[relation_dict[rel]]=rel
        content = json.load(json_file)
        id_arr= [[i,0] for i in range(max(content))]+[[max(content),0]]
        for c in content:
            id_arr[c][1]+=1
        id_arr = sorted(id_arr,key=lambda x:x[1],reverse=True)
        result = id_arr[:50]
        for r in result:
            print('{} {} {}'.format(r[0],relation_arr[r[0]],r[1]))

    pass

def check_cluster_result():
    for e in range(5):
        with open("nyt_relation2id.json", 'r') as json_file:
            relation_dict = json.loads(json_file.read())
        with open("../{}epoch_cluster_result.json".format(e), 'r') as json_file:
            cluster_result = json.loads(json_file.read())
        # sen_file = pd.read_csv('../train_sen_file.csv')
        # print(sen_file)
        with open("../train_data_arr.json", 'r') as json_file:
            train_data_arr,train_data_label = json.loads(json_file.read())
        id2relation =['']*len(relation_dict)
        for k in relation_dict:
            id2relation[relation_dict[k]] = k
        train_data_label_true = [id2relation[item] for item in train_data_label]
        data = pd.DataFrame({'sentence':train_data_arr,
                             'label': train_data_label,
                             'relation': train_data_label_true,
                             '2clusters':cluster_result[0],
                             '3clusters':cluster_result[1],
                             '4clusters':cluster_result[2],
                             '5clusters':cluster_result[3],
                             '6clusters':cluster_result[4],})
        data.to_csv('check_cluster/{}epoch_inspect_cluster_result.csv'.format(e))

    pass

if __name__ == '__main__':
    inspect_category()
    # check_cluster_result()
    pass