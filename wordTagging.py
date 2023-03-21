import nltk
from nltk.corpus import words, stopwords
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


w = WordTagger("dataset1.txt")
print(w.mostCommonWord())
print(w.mostCommonWord(exclusionCode=2))

