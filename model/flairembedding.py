from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings,BertEmbeddings,ELMoEmbeddings

from flair.data import Sentence, Dictionary
from flair.embeddings import WordEmbeddings, CharacterEmbeddings
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle
MAX_SENTENCE_LENGTH = 250
#bert_embedding = BertEmbeddings('/home/user/corpus/pretrain/bert/bert-base-chinese', "-1")
bert_embedding = BertEmbeddings('/home/user/corpus/pretrain/bert/bert-base-uncased', "-1")
#bert_embedding = BertEmbeddings('/home/user/corpus/pretrain/bert/bert-base-german-uncased', "-1")
#bert_embedding = BertEmbeddings('/home/user/corpus/pretrain/bert/bert-base-spanish-wwm-uncased', "-1")

stacked_embeddings = StackedEmbeddings([
    #glove_embedding,
    bert_embedding,
    #elembedding,
    #flair_embedding_forward,
    #flair_embedding_backward,
])
def GenBertEmbeddings(sentence):
    sentembeds = Sentence(sentence)
    stacked_embeddings.embed(sentembeds)
    return sentembeds




def read_instance(input_file, max_sent_length, max_sent_lengthnumber_normalized=True ):
    in_lines = open(input_file,'r').readlines()
    words = []
    wordbert = []
    wordberts = []
    for line in in_lines:
        if len(line.strip()) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            # if number_normalized:
            #     word = normalize_word(word)
            #label = pairs[-5]
            label = pairs[-1]
            words.append(word)
        else:
            if (len(words) > 0) and (len(words) < max_sent_length):
                sentence = " ".join(words)
                sent_embs = GenBertEmbeddings(sentence)
                for token in sent_embs:
                    wordbert.append(token.embedding)
                print(len(wordbert))
                #print(len(ins_bert[idx_sent]), "   ", len(words))
                wordberts.append(wordbert)
            words = []
            wordbert = []
    return wordberts

if __name__ == '__main__':

    # sentence = "Das waren 000 mehr als m Jahr 0000 , wie die Klinik und Poliklinik für Geburtshilfe und Frauenkrankheiten des Klinikums der Johannes Gutenberg- Universität mitteilte ."
    # sent_embs = GenBertEmbeddings(sentence)
    #print(len(sent_embs))
    #train_dir ="/home/user/corpus/NER/conll2002/spanish/train.bioes"
    # dev_dir = "/home/user/corpus/NER/conll2002/spanish/dev.bioes"
    # test_dir = "/home/user/corpus/NER/conll2002/spanish/test.bioes"
    # train_dir = "/home/user/corpus/NER/JNLPBA/train.txt"
    # dev_dir = "/home/user/corpus/NER/JNLPBA/dev.txt"
    # test_dir = "/home/user/corpus/NER/JNLPBA/test.txt"
    #train_dir = "/home/user/corpus/NER/MSRA/msra_train_bio"
    train_dir = "/home/user/corpus/NER/conll2003-en/en/conll2003-9.txt"

    #test_dir = "/home/user/corpus/NER/MSRA/msra_test_bio"
    #test_dir = "/home/user/corpus/NER/broader-twitter/post-f.conll"
    #train_dir = "/home/user/corpus/NER/conll2012/en/ontonotes.train.bioes"
    #dev_dir = "/home/user/corpus/NER/conll2012/en/ontonotes.dev.bioes"
    #test_dir = "/home/user/corpus/NER/conll2012/en/ontonotes.test.bioes"
    # train_dir = "/home/user/corpus/NER/conll2003-ge/NER-de-train-conll-formated.txt"
    # dev_dir = "/home/user/corpus/NER/conll2003-ge/NER-de-dev-conll-formated.txt"
    # test_dir = "/home/user/corpus/NER/conll2003-ge/NER-de-test-conll-formated.txt"
    base_path = "../data/"
    train_bert = read_instance(train_dir,MAX_SENTENCE_LENGTH)
    f = open(base_path+"2003-9.train.bert.bin", 'wb')
    pickle.dump(train_bert, f, 2)
    f.close()

    # dev_bert = read_instance(dev_dir,MAX_SENTENCE_LENGTH)
    # f = open(base_path+"ge.dev.bert.bin", 'wb')
    # pickle.dump(dev_bert, f, 2)
    #f.close()

    # test_bert = read_instance(test_dir, MAX_SENTENCE_LENGTH)
    # f = open(base_path+"twitter.test.bert.bin", 'wb')
    # pickle.dump(test_bert, f, 2)
    # f.close()




# # init embedding
# #glove_embedding = WordEmbeddings('glove')
#
# #elembedding = ELMoEmbeddings("small")
# # init Flair forward and backwards embeddings
# #flair_embedding_forward = FlairEmbeddings('news-forward')
# #flair_embedding_backward = FlairEmbeddings('news-backward')
# # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
#
#
#
# # create sentence.
# sentence = Sentence('The grass is green .')
#
# # embed a sentence using glove.
# stacked_embeddings.embed(sentence)
#
# # now check out the embedded tokens.
# for token in sentence:
#     print(token)
#     print(token.embedding.shape)
#     #print(token.embedding)