from SpacyAnalyzer import SpacyAnalyzer as sa
from SpacyAutoMatcher import SpacyAutoMatcher
import markdown2

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

chatGPT_response = """
Peel and chop the tomatoes, chives and cucumber into very small squares and place in a salad bowl. – Wash, dry and chop the herbs equally and add to the salad bowl. – Let the couscous soak for a few minutes until it becomes fluffy. Then add to the mix. – Pour the oil, add the salt and sprinkle with lemon, then stir everything. – Cover the salad bowl and refrigerate two hours before serving.
"""

stanza_matcher = SpacyAutoMatcher("stanza")
#debug_output = stanza_matcher.debug_for_sentences_as_markdown_table(imperative_sentences)
debug_output = stanza_matcher.debug_for_sentences_as_markdown_table([chatGPT_response])

html_debug_template = """<!doctype html>
<html>
<head>
<title>Action Extractor Debug Output</title>
<style>
table {
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 0.9em;
    font-family: sans-serif;
    width: 100%;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
}

table thead tr {
    background-color: #009879;
    color: #ffffff;
    text-align: left;
}

table th,
table td {
    padding: 12px 15px;
}

table tbody tr {
    border-bottom: 1px solid #dddddd;
}

table tbody tr:nth-of-type(even) {
    background-color: #f3f3f3;
}

table tbody tr:last-of-type {
    border-bottom: 2px solid #009879;
}

table tbody tr.active-row {
    font-weight: bold;
    color: #009879;
}
</style>
</head>
<body>
{html}
</body>
</html>"""

html = markdown2.markdown(debug_output, extras=["tables"])
html = html_debug_template.replace("{html}", html) 

f = open("debug_output.html", "w")
f.write(html)
f.close()
