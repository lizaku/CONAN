import pandas
import random
import numpy as np
import csv
from prettytable import PrettyTable
    

def sample_pandas():
    df = pandas.read_csv('../data/en_hate_original.tsv', sep='\t')
    percent10 = df.shape[0]//10
    rows = np.random.choice(df.index.values, percent10)
    df_10 = df.ix[rows]
    df_90 = df.drop(rows)

    df_10.to_csv('../data/en_hate_test.tsv', sep='\t')
    df_90.to_csv('../data/en_hate_train.tsv', sep='\t')
    
def sample_csv():
    out = {}
    with open('../data/en_hate_original.tsv') as f:
        data = csv.reader(f, delimiter='\t')
        header = next(data)
        for row in data:
            hs = (row[2], row[6])
            cn = (row[3], row[12])
            demo = (row[-6], row[-7], row[-8])
            if hs in out:
                out[hs].append((cn, demo))
            else:
                out[hs] = [(cn, demo)]
    test = {}
    train = {}
    count = 0
    for hs in out:
        if len(out[hs]) <= 4:
            count += 1
            test[hs] = out[hs]
        else:
            train[hs] = out[hs]
    print(count, 'testing examples')
    
    with open('en_hate_test_new.tsv', 'w') as ts:
        ts.write('HS\tCN\tHS_type\tCN_type\tDemographics\n')
        cn_type = {}
        demo_type = {}
        hs_type = {}
        for hs in test:
            hs_t = hs[1]
            try:
                hs_type[hs_t] += 1
            except:
                hs_type[hs_t] = 1
            for cn in test[hs]:
                cn_data = cn[0]
                demo = ' '.join(cn[1])
                try:
                    cn_type[cn_data[1]] += 1
                except:
                    cn_type[cn_data[1]] = 1
                try:
                    demo_type[demo] += 1
                except:
                    demo_type[demo] = 1
                ts.write('\t'.join([hs[0], cn_data[0], hs[1], cn_data[1], demo]) + '\n')
    with open('../seq2seq/data/test.enc', 'w') as f_out:
        for hs in test:
            f_out.write(hs[0] + '\n')
            
    with open('../seq2seq/data/test.dec', 'w') as f_out:
        for hs in test:
            f_out.write(test[hs][0][0][0] + '\n')
                
    
    with open('en_hate_train_new.tsv', 'w') as tr:
        tr.write('HS\tCN\tHS_type\tCN_type\tDemographics\n')
        for hs in train:
            for cn in train[hs]:
                cn_data = cn[0]
                demo = ' '.join(cn[1])
                tr.write('\t'.join([hs[0], cn_data[0], hs[1], cn_data[1], demo]) + '\n')
                
    with open('../seq2seq/data/train.enc', 'w') as f_out:
        for hs in train:
            f_out.write(hs[0] + '\n')
            
    with open('../seq2seq/data/train.dec', 'w') as f_out:
        for hs in train:
            f_out.write(train[hs][0][0][0] + '\n')
    
    cntypes = PrettyTable()
    cntypes.field_names = ["CN type", "Number in test"]
    for cn in cn_type:
        cntypes.add_row([cn, cn_type[cn]])
        
    hstypes = PrettyTable()
    hstypes.field_names = ["HS type", "Number in test"]
    for hs in hs_type:
        hstypes.add_row([hs, hs_type[hs]])
        
    author_types = PrettyTable()
    author_types.field_names = ["Author type", "Number in test"]
    for demo in demo_type:
        author_types.add_row([demo, demo_type[demo]])
        
    print(cntypes)
    print(hstypes)
    print(author_types)
            
            
def dataset_neural():
    out_true = {}
    out_false = {}
    with open('../data/final_hate_dataset_train.csv') as f:
        data = csv.reader(f, delimiter='\t')
        header = next(data)
        data = list(data)
        dic_hs = {}
        for row in data:
            hs = row[1]
            cn = row[4]
            dic_hs[hs] = cn
        for row in data:
            print(row)
            hs = row[1]
            cn = row[4]
            hstype = row[3]
            cntype = row[5]
            false_hs = hs
            i = 0
            while false_hs == hs and i < 3:
                false_hs = random.choice(list(dic_hs.keys()))
                i += 1
                if false_hs != hs:
                    out_false[(false_hs, cn)] = (cn, cntype)
            out_true[(hs, cn)] = (cn, cntype)
    
    with open('../data/final_hate_dataset_neural_no_args.csv', 'w', encoding='utf-8') as f:
        data = csv.writer(f, delimiter=',')
        header = ('Number', 'HS', 'CN', 'IsAnswer', 'CNtype')
        i = 0
        data.writerow(header)
        for hs in out_true:
            h = hs[0]
            cn = out_true[hs][0]
            cntype = out_true[hs][1]
            row = (i, h, cn, 1, cntype)
            i += 1
            data.writerow(row)
            #for cn in out_true[hs]:
            #    row = (i, hs[0] + ' ' + cn[1], cn[0], 1, hs[1], cn[1])
            #    i += 1
            #    data.writerow(row)
        for hs in out_false:
            h = hs[0]
            cntype =  out_false[hs][1]
            cn = out_false[hs][0]
            row = (i, h, cn, 0, cntype)
            data.writerow(row)
            i += 1

if __name__ == '__main__':
    dataset_neural()
