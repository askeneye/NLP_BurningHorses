import oada_an

example = {
    "tokens": ["CNN", "'s", "David", "Ensor", "is", "reporting"],
    "ner_tags": ["B-ORG", "O", "B-PER", "I-PER", "O", "O"],
}

entity_types = ["PER", "LOC", "ORG", "MISC"]

pairs = oada_an.make_oada_pairs(example, entity_types)


print(len(pairs))
print(pairs[0])
print(pairs[1])