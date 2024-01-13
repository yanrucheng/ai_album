
def to_ing_form(verb):
    # Handle verbs ending in 'e'
    if verb.endswith('e') and verb != 'be':
        return verb[:-1] + 'ing'
    # Handle verbs ending in a single vowel followed by a consonant
    elif len(verb) > 2 and verb[-2] in 'aeiou' and verb[-1] not in 'aeiou':
        return verb + verb[-1] + 'ing'
    else:
        return verb + 'ing'

# Sample list of common verbs
common_verbs = ['be', 'have', 'do', 'say', 'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call']  # and so on...

# Create a dictionary with verbs and their '-ing' forms
verb_dict = {verb: to_ing_form(verb) for verb in common_verbs}
l = [to_ing_form(verb) for verb in common_verbs]

from pprint import pprint
pprint(l)
