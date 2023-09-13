from SpacyAnalyzer import SpacyAnalyzer as sa 

analyzer = sa("en_core_web_lg")

#analyzer.show_model_info() 
#analyzer.show_pipeline_info()

#text = "As with dogs from other working breeds, the Australian Cattle Dog is energetic and intelligent with an independent streak. It responds well to structured training, particularly if it is interesting and challenging. It was originally bred to herd by biting, and is known to nip running children. It forms a strong attachment to its owners, and can be protective of them and their possessions. It is easy to groom and maintain, requiring little more than brushing during the shedding period. The most common health problems are deafness and progressive blindness (both hereditary conditions) and accidental injury; otherwise, it is a robust breed with a lifespan of 12 to 16 years."
#doc = analyzer.nlp(text)
#analyzer.show_morph(doc)

# https://universaldependencies.org/u/feat/index.html for morph features standards
# https://en.wikipedia.org/wiki/Imperative_mood

imperative_sentences = [
    "Drive to the store and buy some groceries",
    "Eat the apple if you want",
    "Have a nice trip",
    "Go to your cubicle and finish your work",
    "Hit the ball as hard as you can",
    "Update the modifier and resubmit the claim"
]

#imperative_phrases = analyzer.get_imperative_phrases_for_sentences(imperative_sentences)
#print(*imperative_phrases, sep = "\n")

# gpt prompt: "You are an origami teacher teaching students how to make a paper airplane. Give them all of the steps required to make an airplane."

chatGPT_response = """
1. Start with a rectangular sheet of paper.
2. Fold the top two corners of the paper towards the center.
3. Turn the paper over and fold the top two corners of the paper towards the center again.
4. Flip the paper over again and fold the top of the paper downwards, so that the two flaps created in the previous steps meet in the middle.
5. Fold the paper in half along the line created by the two flaps.
6. Unfold the paper, and then fold the top corners of the paper inwards again.
7. Turn the paper over and fold the top corners of the paper inwards again.
8. Fold the two sides of the paper inwards, so that they meet in the middle.
9. Flip the paper over and fold the two sides of the paper inwards again.
10. Fold the front of the paper downwards, so that the two flaps created in the previous steps meet in the middle.
11. Fold the wings of the paper plane downwards.
12. Your paper plane is now finished!
"""

# what if we build matcher patterns on the fly from exemplary phrases?
# the pattern builder can evolve over time to include wildcards or op hints
patterns = analyzer.build_patterns(imperative_sentences)
actions_from_text = analyzer.get_matching_phrases(patterns, chatGPT_response)
print(*actions_from_text, sep = "\n")

#analyzer.display_parse_dependency(chatGPT_response)
#quit()


#analyzer.show_morph_for_text(chatGPT_response)
#quit()

#actions_from_text = analyzer.get_imperative_phrases_from_text(chatGPT_response)
#print(*actions_from_text, sep = "\n")

#analyzer.show_morph_for_sentences(["Turn the paper over", "fold the top two corners", "Fold the wings of the paper plane downwards"])

#analyzer.show_morph_for_sentences(imperative_sentences)
#analyzer.show_imperative_phrases_for_sentences(imperative_sentences)
