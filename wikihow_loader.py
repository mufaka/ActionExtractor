import os
import pickle
import json
import glob
from SpacyAutoMatcher import SpacyAutoMatcher

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

auto_matcher = SpacyAutoMatcher("stanza")
keys = auto_matcher.build_pattern_keys(wikihow_statements)

with open('stanza_pattern_keys.pkl', 'wb') as f:
    pickle.dump(keys, f)

patterns = []
with open('stanza_pattern_keys.pkl', 'rb') as f:
    patterns = pickle.load(f)
