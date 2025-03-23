from parser import preprocess

SENTENCES = [
    "I am a fuckin goat.",
    "Good muffins cost $3.88\nin New York.  Please buy me... two of them.\n\nThanks.",
    "I'd like to normalize a 6 day * 12 hour workday for men between 16-28",
    "Despite all the self worth voodoo on media, society values you based on your utility.",
    "And when you're young, you have none.",
    "But you have a lot of energy. So trade the energy for experience points & become useful.",
    "Don't learn to love money. Learn to love making money. Inputs > Outputs.",
    "Be strong. Act like a man. 2 Kings 2:2",
]
RETURNS = [
    ["i", "am", "a", "fuckin", "goat"],
    ["good", "muffins", "cost", "in", "new", "york", "please", "buy", "me", "two", "of", "them", "thanks"],
    ["i", "d", "like", "to", "normalize", "a", "day", "hour", "workday", "for", "men", "between"],
    ["despite", "all", "the", "self", "worth", "voodoo", "on", "media", "society", "values", "you", "based", "on", "your", "utility"],
    ["and", "when", "you", "re", "young", "you", "have", "none"],
    ["but", "you", "have", "a", "lot", "of", "energy", "so", "trade", "the", "energy", "for", "experience", "points", "become", "useful"],
    ["don", "t", "learn", "to", "love", "money", "learn", "to", "love", "making", "money", "inputs", "outputs"],
    ["be", "strong", "act", "like", "a", "man", "kings"]
]




def test_sentence():
    for pos, sentence in enumerate(SENTENCES):
        processed = preprocess(sentence)
    assert processed == RETURNS[pos]
