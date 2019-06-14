import json
import csv
import random

PATH = '../data/EN_hate_speech_dataset_final.json'

with open(PATH) as f_in:
    data = json.loads(f_in.read())

with open('../data/final_hate_dataset.csv', 'w', encoding='utf-8') as f_out:
    f = csv.writer(f_out, delimiter='\t')
    f.writerow(['Number', 'OrigHS', 'Paraphrase1', 'Paraphrase2', 'HStype', 'HSsubtype', 'CN', 'CNtype'])
    number = 1
    data_dic = {}
    for elem in data:
        orighs = elem['originalHateSpeechEN']
        try:
            par1 = elem['paraphrasedHateSpeechEN1']
        except KeyError:
            par1 = ''
        try:
            par2 = elem['paraphrasedHateSpeechEN2']
        except KeyError:
            par2 = ''
        hstype = elem['hsType']
        hssubtype = elem['hsSubType']
        cn = elem['originalCounterSpeechEN']
        cntype = elem['cnType']
        if orighs in data_dic:
            data_dic[orighs].append(cn)
        else:
            data_dic[orighs] = [cn]
        row = [str(number), orighs, par1, par2, hstype, hssubtype, cn, cntype]
        number += 1
        f.writerow(row)

test_hs = {}
repl = 0
for hs in data_dic:
    repl += len(data_dic[hs])
    if len(data_dic[hs]) < 6:
        test_hs[hs] = random.choice(data_dic[hs])
print('average', repl/len(data_dic))
        
train_out = open('../data/final_hate_dataset_train.csv', 'w', encoding='utf-8')
test_out = open('../data/final_hate_dataset_test.csv', 'w', encoding='utf-8')
header = ('Number', 'HS', 'HStype', 'HSsubtype', 'CN', 'CNtype')
train_csv = csv.writer(train_out, delimiter='\t')
train_csv.writerow(header)
test_csv = csv.writer(test_out, delimiter='\t')
test_csv.writerow(header)
num_train = 1
num_test = 1
processed_test = set()
for elem in data:
    orighs = elem['originalHateSpeechEN']
    try:
        par1 = elem['paraphrasedHateSpeechEN1']
    except KeyError:
        par1 = ''
    try:
        par2 = elem['paraphrasedHateSpeechEN2']
    except KeyError:
        par2 = ''
    hstype = elem['hsType']
    hssubtype = elem['hsSubType']
    cn = elem['originalCounterSpeechEN']
    cntype = elem['cnType']
    if orighs in test_hs and orighs not in processed_test:
        row = [str(num_test), orighs, hstype, hssubtype, test_hs[orighs], cntype]
        test_csv.writerow(row)
        num_test += 1
        processed_test.add(orighs)
    elif orighs not in test_hs:
        row1 = [str(num_train), orighs, hstype, hssubtype, cn, cntype]
        train_csv.writerow(row1)
        num_train += 1
        row2 = [str(num_train), par1, hstype, hssubtype, cn, cntype]
        train_csv.writerow(row2)
        num_train += 1
        row3 = [str(num_train), par2, hstype, hssubtype, cn, cntype]
        train_csv.writerow(row3)
        num_train += 1
train_out.close()
test_out.close()

