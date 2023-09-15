import os
import pickle
from SpacyAnalyzer import SpacyAnalyzer as sa
from SpacyAutoMatcher import SpacyAutoMatcher
import json
import glob

#analyzer.show_model_info() 
#analyzer.show_pipeline_info()

#text = "As with dogs from other working breeds, the Australian Cattle Dog is energetic and intelligent with an independent streak. It responds well to structured training, particularly if it is interesting and challenging. It was originally bred to herd by biting, and is known to nip running children. It forms a strong attachment to its owners, and can be protective of them and their possessions. It is easy to groom and maintain, requiring little more than brushing during the shedding period. The most common health problems are deafness and progressive blindness (both hereditary conditions) and accidental injury; otherwise, it is a robust breed with a lifespan of 12 to 16 years."
#doc = analyzer.nlp(text)
#analyzer.show_morph(doc)

# https://universaldependencies.org/u/feat/index.html for morph features standards
# https://en.wikipedia.org/wiki/Imperative_mood

imperative_sentences = [
    "Drive to the store and buy some groceries",
    "Drive to the store",
    "Eat the apple if you want",
    "Eat the apple",
    "Have a nice trip",
    "Go to your cubicle and finish your work",
    "Go to your cubicle",
    "finish your work",
    "Hit the ball as hard as you can",
    "Hit the ball",
    "Update the modifier and resubmit the claim",
    "Fold the top of the paper downwards",
    "Fold the top two corners of the paper"
]

#auto_matcher = SpacyAutoMatcher("en_core_web_trf")
#keys = auto_matcher.build_pattern_keys(imperative_sentences)
#print(keys)
#quit()


#imperative_phrases = analyzer.get_imperative_phrases_for_sentences(imperative_sentences)
#print(*imperative_phrases, sep = "\n")

# gpt prompt: "You are an origami teacher teaching students how to make a paper airplane. Give them all of the steps required to make an airplane."

chatGPT_response = """
Peel and chop the tomatoes, chives and cucumber into very small squares and place in a salad bowl. – Wash, dry and chop the herbs equally and add to the salad bowl. – Let the couscous soak for a few minutes until it becomes fluffy. Then add to the mix. – Pour the oil, add the salt and sprinkle with lemon, then stir everything. – Cover the salad bowl and refrigerate two hours before serving.
"""

# what if we build matcher patterns on the fly from exemplary phrases?
# the pattern builder can evolve over time to include wildcards or op hints
#patterns = analyzer.build_patterns(imperative_sentences)
#print(patterns)

# can we load the wikiHow data?
# /mnt/d/ML/Datasets/WIKIHOW/archive/wikiHow0.json

'''
with open('action_patterns.pkl', 'rb') as f:
    patterns = pickle.load(f)
    first_100 = patterns[:100]

    for pattern in first_100:
        print(pattern)
'''

# too many patterns means slow AF matching in spaCy ...
# [{'TAG': 'VB', 'OP': '+'}, {'POS': 'ADP', 'OP': '{1}'}, {'POS': 'DET', 'OP': '{1}'}, {'TAG': 'NN', 'OP': '+'}, {'POS': 'ADP', 'OP': '{1}'}, {'POS': 'DET', 'OP': '{1}'}, {'POS': 'ADJ', 'OP': '{1}'}, {'TAG': 'NN', 'OP': '+'}, {'POS': 'PUNCT', 'OP': '{1}'}]

'''
wikihow_statements = []

wikihow_glob = f'/mnt/d/ML/Datasets/WIKIHOW/archive/wikiHow*.json'
wikihow_files = glob.glob(wikihow_glob)

for file in wikihow_files:
    with open(file) as wikiHow:
        data = json.load(wikiHow)
        for article in data:
            if "Parts" in article:
                for parts in article["Parts"]:
                    if "steps" in parts:
                        for step in parts["steps"]:       
                            if "Headline" in step:
                                wikihow_statements.append(step["Headline"])

auto_matcher = SpacyAutoMatcher("en_core_web_trf")
keys = auto_matcher.build_pattern_keys(wikihow_statements)

with open('action_pattern_keys.pkl', 'wb') as f:
    pickle.dump(keys, f)

patterns = []
with open('action_pattern_keys.pkl', 'rb') as f:
    patterns = pickle.load(f)
'''
stanza_matcher = SpacyAutoMatcher("stanza")
stanza_matches = stanza_matcher.get_imperative_phrases(chatGPT_response)
print(stanza_matches)

quit()

auto_matcher = SpacyAutoMatcher("en_core_web_trf", "wikihow_action_pattern_keys.pkl")
matches = auto_matcher.get_matching_phrases_by_key(chatGPT_response)
print("---- TEXT MATCHES ----")
print(matches)
print()

sentence_matches = auto_matcher.get_matching_sentence_by_key(imperative_sentences)
print("---- SENTENCE MATCHES ----")
print(sentence_matches)



#actions_from_text = auto_matcher.get_matching_phrases(wikihow_statements, chatGPT_response)
#print(*actions_from_text, sep = "\n")

#analyzer = sa("en_core_web_lg")
#actions_from_text = analyzer.get_matching_phrases(imperative_sentences, chatGPT_response)
#print(*actions_from_text, sep = "\n")

#analyzer.display_parse_dependency(chatGPT_response)
#quit()

#analyzer.show_morph_for_text(chatGPT_response)
#quit()

#actions_from_text = analyzer.get_imperative_phrases_from_text(chatGPT_response)
#print(*actions_from_text, sep = "\n")

#analyzer.show_morph_for_sentences(["Turn the paper over", "fold the top two corners", "Fold the wings of the paper plane downwards"])

#analyzer.show_morph_for_sentences(imperative_sentences)
#analyzer.show_imperative_phrases_for_sentences(imperative_sentences)
