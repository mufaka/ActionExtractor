import spacy 
from spacy.matcher import Matcher
import pandas as pd 

'''
Analyze the annotated data in the following to, hopefully, discover patterns in
nlp pos-tagging that can be useful in extracting actions from text.

"TV-AfD: An Imperative-Annotated Corpus from The Big Bang Theory and Wikipedia’s 
Articles for Deletion Discussions"

Citation
When using this dataset for any academic work or publication, please cite the following paper:

Yimin Xiao, Zong-Ying Slaton, and Lu Xiao. 2020. TV-AfD: An Imperative-Annotated Corpus from 
The Big Bang Theory and Wikipedia’s Articles for Deletion Discussions. In Proceedings of The 
12th Language Resources and Evaluation Conference, European Language Resources Association, 
Marseille, France, 6544–6550. Retrieved from https://www.aclweb.org/anthology/2020.lrec-1.805

NOTE: Parakeet Lab’s Email Data Set3 which was originally built
from Enron Email Corpus2 has data labeled for intention.
The dataset has in total 5,204 labeled sentences and 1,908
of them are positive cases. The targeted intention in the
sentence includes the request intention and the propose
intention (Cohen, Carvalho and Mitchell, 2004), which is
similar to some of the functions of imperative sentences. 

NOTE: The English Web Treebank (Ann Bies et al. 2012) contains
data from English blogs, news, emails, reviews, and
question-answer pairs, covering both formal and informal
texts. All sentences in this treebank are POS tagged.
Imperative sentences are labeled using a “Mood-Imp' tag
on the verb in sentences. There are in total of 944
imperative sentences out of 12,543 sentences from the
dataset.
'''

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# try reading from github directly
url = "https://raw.githubusercontent.com/yiminxsisu/TV-AfD_Imperative_Corpus/master/TV_show/Classified_data/season_2_labeled/finallabeldata2x01.txt"

'''
The columns respectively are: 
    data source (in the formant of nxmm with 'n' indicating the season of the show and 'mm' representing the episode), 
    text, 
    imperative classification (1 for imperative and 0 for non-imperative), 
    imperative category, 
    and whether the imperative sentence has affixal negative markers or not. (ni = negative markers included, nf = negative markers-free)

    need to find documentation on imperative category and affixal negative markers

'''
col_names = ["source", "text", "is_imperative", "category", "has_neg_mark", "ext1"] # some lines have extraneous field(s)
df = pd.read_csv(url, sep="\t", names=col_names)

#print_full(df)
#quit()

#df.reset_index()

imperative_sentences = []

for index, row in df.iterrows():
    # marked imperative?
    if (row["is_imperative"] == 1 and row["has_neg_mark"] == "nf"):
        imperative_sentences.append(f'{row["category"]}-{row["text"]}')

print(*imperative_sentences, sep = "\n")

# print_full(df)

