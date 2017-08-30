

# Logic used
# P(class|text)=P(class)*p(word1|class)*p(word2|class).....*p(wordN|class)/p(word1)*p(word2)....*p(wordN)
# the topics are the classes and words extracted from
# the test file(using words_in _file functions:refer report) are considered for classification.
#Words containg letters are removed
# Training involves filling 3 lists and a counter
#     pTt=>  probability for each topic
#     wgt=>no of words dictionary for each topic
#     all_words=> no of words dictionary
#     total=0, total words
#   They above are stored using pickle
#   Distinctive words: freq variable used as limiter for extreamly rare words
#       for every word in topic its corresponding p(topic|word) proportional to p(word|topic)/p(word) is calculated
#       Max 10 are selected for each topic
#       put them in a file
#Testing involves going through each file
    #Generating words using words_in _file function
    #Maintaing a list of probabilities of all 20 labels given the textfile(words)
    #P(class|text)
    #2 parameters to filter extreamly rare words and insignificant words
    #classify to the folder with max probability in the list and increment the confusion matrix



#python 3 code

import sys
import pickle
import os
import string
import numpy as np
from operator import itemgetter
from collections import defaultdict


save_dir=os.getcwd()
mode = str(sys.argv[1])
directory = str(sys.argv[2])
# os.chdir("/home/aditya/PycharmProjects/AI4/part1/train")    # set path
# os.chdir("/home/aditya/PycharmProjects/AI4/part1/test")     # fix testing path
file_name = str(sys.argv[3])
file_name += ".p"


'''Takes a file and gives all words in it'''
def words_in_file1(text):#gives better words
    output = []
    lines = text.readlines()
    for each_line in lines:
        words_split = each_line.lower().split()
        output += words_split
    return [word for word in output if word.isalpha()]

def words_in_file2(text):#gives better accuracy,default
    output = []
    lines = text.readlines()
    for each_line in lines:
        words_split = each_line.lower().translate(str.maketrans('', '', string.punctuation)).split()
        output += words_split
    return [word for word in output if word.isalpha()]


if mode=="train":

    '''Training'''

    os.chdir(directory)

    print ("Training started")

    no_files={}                                                             #will contain proportion of all folders
    wgt=defaultdict(int)                                                    #no of words dictionary for each topic
    all_words=defaultdict(int)                                              ##no of words dictionary
    total=0


    for folder in os.listdir(os.getcwd()):                                  #iterating through folders
        if os.path.isdir(folder):
            z=os.listdir(os.path.join(os.getcwd(), folder))
            no_files[folder]=len(z)                                         #folder length
            in_folder = defaultdict(int)                                    #dict of words in the folder
            for file in z:                                                  #iterating through files
                with open(os.path.join(os.getcwd(),folder,file), encoding="ISO-8859-1") as f:
                    words=words_in_file2(f)                                  #fetching words in a file
                    for word in words:                                      #word saved in all_words and in_folder
                        total+=1
                        all_words[word] += 1
                        in_folder[word]+=1
            wgt[folder]=in_folder

    #getting total and genrating p(Ti)
    s=sum(no_files.values())
    pTt={k:v/s for k,v in no_files.items()}

    print ("Accessed folders")

    "-------------------------------------------------------------------------------------------------------------------"

    '''Storing file'''

    print ("Creating model-file")
    ##saving to a file
    afile = open(os.path.join(save_dir, file_name), 'wb')
    pickle.dump((pTt,wgt,all_words,total), afile)
    afile.close()
    print ("created file")


    "-------------------------------------------------------------------------------------------------------------------"


    '''top 10 words for each topic'''

    print ('Generating words')
    to_write='The top 10 words are as follows:'                      #string that stores all the generated words
    freq = 0.0005
    for topic in wgt:                                               #iterating through topics in wgt
        line="\nFor "+str(topic)+':\n'
        to_write+= line
        find_max=[]                                                 #stores (value,word) for each word

        num = pTt[topic]                                            #p(T=t) given by pTt,ie pTt[topic]
        wtopic= sum(wgt[topic].values())                            #words in topic

        for word in wgt[topic]:                                     #find p(t|w) for all words
            pwt = wgt[topic][word]/wtopic                           #p(w|t)
            if pwt> freq:                                         #limiter for extreamly rare words
                pw= all_words[word]/s                               #probability of word
                find_max.append((num*pwt/pw, word))

        find_max.sort()
        temp2=find_max[-10:]                                        #get top 10 with highest probability
        temp=[b for a,b in temp2]                                   #add to string
        to_write+=" ".join(temp)


    '''put them in file'''

    w= open(os.path.join(save_dir, "distinctive_words.txt"),'w')
    w.write(to_write)
    w.close()

    print ("Words saved to file")

"-------------------------------------------------------------------------------------------------------------------"

if mode == "test":

    '''Loading model'''
    os.chdir(save_dir)
    # reload object from file
    file2 = open(file_name, 'rb')
    new_d = pickle.load(file2)
    pTt = new_d[0]
    wgt = new_d[1]
    all_words = new_d[2]
    total = new_d[3]
    file2.close()
    print ("Model fetched")


    "-------------------------------------------------------------------------------------------------------------------"


    '''Testing '''

    os.chdir(directory)

    print ("Testing started,takes some time")
    trix = np.zeros([20, 20], dtype=int)            #empty 20*20 matrix


    for folder in os.listdir(os.getcwd()):          #iterating through test folders
        if os.path.isdir(folder):
            for file in os.listdir(os.path.join(os.getcwd(), folder)):      #test files
                with open(os.path.join(os.getcwd(),folder,file), encoding="ISO-8859-1") as f:
                    prob = []                                  #probabilities of all labels for a documnent are appended here
                    wrds=words_in_file2(f)                      #gets words in doc

                    for label in os.listdir(os.getcwd()):               #for all labels
                        words_folder = sum(wgt[label].values())         #total words in topic
                        ptopic = pTt[label]                             #probability of topic
                        temprob=1                                       #itialized to 1,probabilities multiplied in this label
                        for word in wrds:
                            if word in all_words:                       #for all recognized words
                                pgivent=wgt[label][word]/words_folder   #p(word|topic)
                                if pgivent > 0.000001:                   #filters extreamly rare words
                                    pofw = all_words[word]/total        #p(word)
                                    if pofw < 0.001:                   #filters insignificant words
                                        temprob=temprob*pgivent/pofw
                        #calculates p for a label and appends
                        prob.append((temprob*ptopic,label))
                classify=max(prob,key=itemgetter(0))[1]             #selects the topic with the max p from the list
                trix[os.listdir(os.getcwd()).index(folder)][os.listdir(os.getcwd()).index(classify)]+=1     #increments confusion matrix

    print ("Confusion matrix:")
    print(trix)
    accuracy= np.trace(trix)/trix.sum()*100
    print("Accuracy:",accuracy)


