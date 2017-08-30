# Sentence-Generator-and-Grammar-Checker


Generated novel sentences by sampling from Markov chain using text from Brown corpus.
Developed a grammar checker that will suggest replacing words if wrongly used in the sentence.

The dataset of documents is organized into twenty broad topics (ex: religion, finance, etc.). 
Consider dataset of documents D = {d 1 , d 2 , ..., d n }, a set of topics T = {t 1 , t 2 , ..., t m }, and a set of words in the English language W = {w 1 , w 2 , ..., w o }. A given document D ∈ D in corpus has a topic T ∈ T and a set of words W ⊆ W. In training set, we’ll always be able to observe D, W and T . In the test set, we’ll always be able to observe D and W but never T. The corpus can be modeled as a Bayes Net with random variables D, T and {W 1 , ..., W o }, where W i ∈ {0, 1} indicates whether word w i ∈ W appears in the document or not. Variable D is connected to T via a directed edge, and then T is connected to each W i via a directed edge.
