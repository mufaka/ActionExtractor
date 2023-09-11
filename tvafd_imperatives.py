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
    text, imperative classification (1 for imperative and 0 for non-imperative), 
    imperative category, 
    and whether the imperative sentence has affixal negative markers or not.

    ParserError: Error tokenizing data. C error: Expected 3 fields in line 7, saw 6 <-- docs say 4 max
'''
col_names = ["source", "text", "category", "has_neg_mark", "ext1", "ext2"]
df = pd.read_csv(url, sep="\t", names=col_names)
print_full(df)
