# For each programming problem, please include a detailed comments section at the top of your code that
# describes: (1) a description of how you formulated the problem, including precisely defining the abstrac-
# tions; (2) a brief description of how your program works; (3) a discussion of any problems, assumptions,
# simplifications, and/or design decisions you made; and (4) answers to any questions asked below in the
# assignment.

#location set in 2 places,that needs to be changed
#In your report, show the confusion matrix and report the overall classification accuracy.


import os
import string
# import numpy as np
from operator import itemgetter
from collections import defaultdict

#train

def unique_words(text):
    output = []
    lines = text.readlines()
    for each_line in lines:
        words_split = each_line.lower().split()
        # words_split = each_line.lower().translate(str.maketrans('', '', string.punctuation)).split()
        output += words_split
    # return {word for word in output if word.isalpha()}
    return output


os.chdir("/home/aditya/PycharmProjects/AI4/part1/train")
no_files={}
wgt=defaultdict(int)
all_words={}

def all_words_filler(wrd):
    global all_words
    if wrd not in all_words:
        all_words[wrd] = 1
    else:
        all_words[wrd] += 1


for folder in os.listdir(os.getcwd()):
    if os.path.isdir(folder):
        z=os.listdir(os.path.join(os.getcwd(), folder))
        no_files[folder]=len(z)
        x = defaultdict(int)
        for file in z:
            with open(os.path.join(os.getcwd(),folder,file), encoding="ISO-8859-1") as f:
                words=unique_words(f)
                for word in words:
                    all_words_filler(word)
                    if word not in x :
                        x[word]=1
                    else:
                        x[word]+=1
        wgt[folder]=x


s=sum(no_files.values())
pTt={k:v/s for k,v in no_files.items()}


"-------------------------------------------------------------------------------------------------------------------"
#
# for field in wgt:
#     wgt[field]={k: wgt[field][k] for k in wgt[field] if wgt[field][k] in sorted(wgt[field].values())[50:]}

"-------------------------------------------------------------------------------------------------------------------"

# #removing stopwords, can use in for loop, consider individual as well:university
# rep=sorted(all_words.values())[-50:]
# for word in all_words:
#     if all_words[word] in rep:
#         print(word)

"-------------------------------------------------------------------------------------------------------------------"

# ##saving to a file
# import pickle
# afile = open(r'model-file', 'wb')
# pickle.dump((topics,pTt,wgt,all_words), afile)
# afile.close()
#
# #reload object from file
# file2 = open(r'model-file', 'rb')
# new_d = pickle.load(file2)
# file2.close()
# # print(new_d)

"-------------------------------------------------------------------------------------------------------------------"

#top 10 words for each topic

to_write='The top 10 words are as follows:'
freq = 0.01
for topic in wgt:
    line="\nFor "+str(topic)+':\n'
    to_write+= line
    find_max=[]
    d=no_files[topic]
    # p(T=t) given by pTt,ie num1=pTt[topic] ignored
    num = pTt[topic]
    for word in wgt[topic]:
        if wgt[topic][word] > freq:
            # pw= all_words[word]/s
            # pwt = wgt[topic][word] / d
            pw = 0        # words_split = each_line.lower().translate(str.maketrans('', '', string.punctuation)).split()

            for other_topic in wgt:
                pw += max(1E-6, wgt[other_topic][word])
            pwt= max(1E-6, wgt[topic][word])                     # p(w|t)=wgt[topic][word]/no_files[topic]
            find_max.append((pwt/pw,word))              #*num
    find_max.sort()
    temp2=find_max[-10:]
    temp=[b for a,b in temp2]
    to_write+=" ".join(temp)

print(to_write)
#
# # put them in file
#
# w= open("distinctive_words.txt",'w')
# w.write(to_write)
# w.close()


# "-------------------------------------------------------------------------------------------------------------------"


# test

#
# os.chdir("/home/aditya/PycharmProjects/AI4/part1/test")
# trix = np.zeros([20, 20], dtype=int)
#
# for folder in os.listdir(os.getcwd()):
#     if os.path.isdir(folder):
#         # print(os.listdir(os.path.join(os.getcwd(), folder)))
#         ptopic=pTt[folder]
#         for file in os.listdir(os.path.join(os.getcwd(), folder)):
#             with open(os.path.join(os.getcwd(),folder,file), encoding="ISO-8859-1") as f:
#                 prob = []
#                 uni=unique_words(f)
#
#                 for label in os.listdir(os.getcwd()):  #check from here
#                     temprob=1
#                     for word in uni:
#                         if word in all_words:
#                             pofw=all_words[word]
#                             pgivent = wgt[label].get(word,0.00000000000001)/no_files[label]
#                             temprob*=ptopic*pofw/pgivent
#                     prob.append((temprob,label))
#                 classify=max(prob,key=itemgetter(0))[1]
#                 trix[os.listdir(os.getcwd()).index(folder)][os.listdir(os.getcwd()).index(classify)]+=1
# print(trix)
# accuracy= np.trace(trix)/trix.sum()*100
# print(accuracy)
