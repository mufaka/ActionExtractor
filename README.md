# ActionExtractor
Research on extracting actions from text.

## The best / easiest algorithm found so far
Find nouns that have a first ancestor that is a verb in the imperative mood and then join children into a phrase. This is demonstrated in spaCy_learning.py which uses SpacyAutoMatcher.py with Stanza. Stanza is the only NLP model I've found that tags the correct morphological features of verbs. A lot more testing is needed but this is the approach I am going forward with. An example debug output follows.

![image](https://github.com/mufaka/ActionExtractor/assets/8632538/22af9042-e4f6-4380-999d-a9ade4d5adb9)


## Citation - Not currently using but the code is still in the repository so leaving here
Yimin Xiao, Zong-Ying Slaton, and Lu Xiao. 2020. TV-AfD: An Imperative-Annotated Corpus from The Big Bang Theory and Wikipedia’s Articles for Deletion Discussions. In Proceedings of The 12th Language Resources and Evaluation Conference, European Language Resources Association, Marseille, France, 6544–6550. Retrieved from https://www.aclweb.org/anthology/2020.lrec-1.805
