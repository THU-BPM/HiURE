import json

if __name__ == '__main__':
    # not finished yet
    prefix = 'data/fewRel/'
    data_set = ['train','val']
    for d in data_set:
        with open(prefix + 'data/{}_wiki.json'.format(d), 'r') as f:
            line = f.readline()
            data = json.loads(line)

            sentence = []
            label = []
            label_counter=0

            print(len(data))
            for key in data:
                for sen in data[key]:
                    tokens = sen["tokens"]
                    label.append(label_counter)
                    h_index = sen['h'][2][0]
                    t_index = sen['t'][2][0]
                    if h_index[0]>t_index[0]:
                        tmp = h_index
                        h_index=t_index
                        t_index=tmp
                    tokens.insert(h_index[0], '<e1>')
                    tokens.insert(h_index[-1]+2, '</e1>')
                    tokens.insert(t_index[0]+2, '<e2>')
                    tokens.insert(t_index[-1]+4, '</e2>')
                    new_sen = ' '.join(tokens)
                    sentence.append(new_sen)
                label_counter+=1
            with open(prefix + 'data/{}_sentence.json'.format(d), 'w') as fw:
                json.dump(sentence, fw, indent=4)
            with open(prefix + 'data/{}_label_id.json'.format(d), 'w') as fw:
                json.dump(label, fw, indent=4)
