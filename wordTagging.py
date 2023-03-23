import nltk
from nltk.corpus import words, stopwords
import string
import gensim

def cleaned_sentence(s):
        clean_sentence = []
        for w in s:
            if w not in stopwords.words("english") and w not in set(string.punctuation):
                clean_sentence.append(w.lower())
        return clean_sentence

class WordTagger:
    def __init__(self, filename):
        self.filename = filename
        with open (filename, 'r', errors="replace") as f:
            self.lineList = f.readlines()
            f.close()

        self.tokenizedList = []
        self.taggedList = []

        #Tokenization
        for i in self.lineList:
            self.tokenizedList.append(nltk.word_tokenize(i))
        #Tagging
        for i in self.tokenizedList:
            self.taggedList.append(nltk.pos_tag(i))

        #print(self.taggedList[0][0])

    def createFreqDist(self):
        fDist = nltk.FreqDist()
        for s in self.tokenizedList:
            for word in s:
                fDist[word] += 1
        return fDist
    def mostCommonToken(self):
        fd = self.createFreqDist()
        return(fd.most_common()[0][0])

    def mostCommonWord(self, exclusionCode=0):
        #exclusionCode: 0=none, 1=stop words, 2=non-nouns
        fd = self.createFreqDist()
        for i in range(len(fd.most_common())):
            if fd.most_common()[i][0] not in words.words():
                continue
            if exclusionCode==1 and (fd.most_common()[i][0] in stopwords.words("english")):
                continue
            if exclusionCode==2 and nltk.pos_tag([fd.most_common()[i][0]])[0][1][:2] != "NN":
                    continue
            else:
                return (fd.most_common()[i][0])

    def latentDirichletAllocation(self):
        dirty_words = []
        clean_words = []
        for i in self.tokenizedList:
            dirty_words.append(i)
        for s in dirty_words:
            clean_words.append(cleaned_sentence(s))
        dictionary = gensim.corpora.Dictionary(clean_words)
        DT_matrix = [dictionary.doc2bow(sentence) for sentence in clean_words]
        Lda_object = gensim.models.ldamodel.LdaModel
        lda_model_1 = Lda_object(DT_matrix, num_topics=3, id2word=dictionary)
        return("LDA Model Results: " + str(lda_model_1.print_topics(num_topics=3, num_words=5)))
        

    def topic(self):
        return(self.filename + "\nBag of words: " + self.mostCommonWord(exclusionCode=1) + "\nMost Common Noun: " + self.mostCommonWord(exclusionCode=2) + "\n" + self.latentDirichletAllocation())


w = WordTagger("dataset4.txt")
print(w.topic())