from operator import itemgetter
from keras.models import load_model
from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd

# model with arguments
# model = load_model('./checkpoints/1559657668/lstm_50_50_0.17_0.25.h5') # 0.0833 seq_length=10, emb_dim=50, drop=0.17
# model = load_model('./checkpoints/1559660198/lstm_50_50_0.10_0.25.h5') # 0.1042 seq_length=50, emb_dim=100, drop=0.1
# model = load_model('./checkpoints/1559660786/lstm_50_50_0.10_0.25.h5') # 0.0625 seq_length=100, emb_dim=200, drop=0.1
# model = load_model('./checkpoints/1559661637/lstm_50_50_0.10_0.25.h5') # 0.0833 seq_length=100, emb_dim=100, drop=0.1
# model = load_model('./checkpoints/1559662284/lstm_50_50_0.10_0.25.h5') # 0.0208 seq_length=50, emb_dim=50, drop=0.1
model = load_model('./checkpoints/1560474743/lstm_50_50_0.10_0.10.h5') # seq_length=50, drop=0.1, lstm_drop=0.1
model = load_model('./checkpoints/1560475412/lstm_50_50_0.25_0.25.h5') # seq_length=50, drop=0.25, lstm_drop=0.25
# 3 negative examples
#model = load_model('./checkpoints/1560122132/lstm_50_50_0.10_0.25.h5')
# model without arguments
#model = load_model('./checkpoints/1559659364/lstm_50_50_0.17_0.25.h5')


df = pd.read_csv('../data/final_hate_dataset_test.csv', sep='\t')
hs_sentences = list(df['HS'])
cn_sentences = list(df['CN'])
cntype = list(df['CNtype'])
# remove the cn type
# cntype = ['none' for x in cntype]

top_n = 3

tokenizer, embedding_matrix = word_embed_meta_data(hs_sentences + cn_sentences,  siamese_config['EMBEDDING_DIM'])

guess = 0
for i in range(len(hs_sentences)):
    resp = cn_sentences[i]
    hs = hs_sentences[i]
    all_pairs = [(hs, cn) for cn in cn_sentences]
    test_data_x1, test_data_x2, leaks_test, cntypes = create_test_data(tokenizer, all_pairs, cntype, siamese_config['MAX_SEQUENCE_LENGTH'])

    preds = list(model.predict([test_data_x1, test_data_x2, leaks_test, cntypes], verbose=1).ravel())
    results = [(x, y, z) for (x, y), z in zip(all_pairs, preds)]
    results.sort(key=itemgetter(2), reverse=True)
    top = results[:top_n]
    responses = [x[1] for x in top]
    if resp in responses:
        print('Guess')
        guess += 1
    print(hs, resp, top)
print('Accuracy', float(guess)/len(hs_sentences))
