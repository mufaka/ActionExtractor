from SpacyAnalyzer import SpacyAnalyzer as sa 

analyzer = sa("en_core_web_lg")

analyzer.show_model_info() 
analyzer.show_pipeline_info()

text = "As with dogs from other working breeds, the Australian Cattle Dog is energetic and intelligent with an independent streak. It responds well to structured training, particularly if it is interesting and challenging. It was originally bred to herd by biting, and is known to nip running children. It forms a strong attachment to its owners, and can be protective of them and their possessions. It is easy to groom and maintain, requiring little more than brushing during the shedding period. The most common health problems are deafness and progressive blindness (both hereditary conditions) and accidental injury; otherwise, it is a robust breed with a lifespan of 12 to 16 years."
doc = analyzer.nlp(text)
analyzer.show_morph(doc)

# https://universaldependencies.org/u/feat/index.html for morph features standards
# https://en.wikipedia.org/wiki/Imperative_mood
'''
imperative_sentences = [
    "Drive to the store.",
    "Eat the apple if you want.",
    "Have a nice trip!",
    "I have to ask you to stop.",
    "It would be great if you made us a drink.",
    "Go to your cubicle!",
    "Hit the ball."
]
'''
#show_morph_for_sentences(imperative_sentences)
#show_imperative_phrases_for_sentences(nlp, imperative_sentences)
