import json
import re
from collections import Counter
import pandas as pd
import json
import ast
import unicodedata
import inflect
p = inflect.engine()




CSV_PATH = "data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
OUT_PATH = "data/processed/ingredient_vocab.json"
TOP_K = 1000
MIN_FREQ = 5   # you can tune this (3–10)


BAD_WORDS = set([
# measurements
"cup","cups","tbsp","tablespoon","tablespoons","tsp","teaspoon","teaspoons",
"oz","ounce","ounces","lb","pound","pounds","gram","grams","kg","ml","liter","liters",
"quart","quarts","pint","pints","gallon","gallons","drop","drops","pinch","pinches",
"inch", "inchthick", "lengthwise", "crosswise", "round",


# numbers
"one","two","three","four","five","six","seven","eight","nine","ten","half","quarter",
"double","triple","dozen","¼","½","¾","1/2","1/4","3/4",

# tools / containers
"pan","pot","bowl","knife","fork","spoon","whisk","skillet","grill","oven","tray",
"plate","dish","jar","can","cans","bottle","bag","bags","box","boxes","tin",
"ramekin","mixer","processor","blender","grater","peeler","board","rack","sheet",
"ramekin", "cheesecloth", "mandoline", "bowl", "spoon", "pan", "scissors",

# actions / preparation
"chopped","minced","sliced","diced","crushed","ground","grated","shredded",
"peeled","seeded","cored","trimmed","halved","quartered","cut","divided",
"boiled","fried","baked","roasted","grilled","steamed","sauteed","smoked",
"mixed","tossed","folded","whisked","blended","marinated","soaked","drained",
"rinsed","washed","patted","cooled","heated","softened","melted",
"thinly", "finely", "toasted", "chilled", "pitted", "crumbled", "beaten", "shaved", "thawed",


# descriptors
"fresh","freshly","large","small","medium","extra","virgin","organic","raw","ripe",
"dry","wet","hot","cold","warm","thin","thick","fine","coarse","whole","broken",
"soft","hard","light","dark","sweetened","unsweetened","boneless","skinless",

# shapes / parts
"slices","slice","chunks","chunk","pieces","piece","cubes","wedges","rings",
"strips","leaves","tops","roots","stems","skins","shells","bones","pits",

# metadata / noise
"optional","taste","desired","needed","serve","serving","recipe","note","notes",
"see","use","using","plus","more","about","each","per","approx","approximately",
"etc","and","or","of","to","with","for","on","in","into","by","from","as","is","are","be",

# cooking techniques
"julienne","julienned","braised","braising","poached","blanched","grinding",
"grilling","roasting","frying","stirring","seasoning","garnishing","decorating",

# non-food noise
"ikea","disco","masking","doublesided","animalshaped","leafshape","individualsize",
"combination","northwest","holiday","umbrellas","vases","luster","quartt","widetooth"
])

BAD_SUFFIXES = (
    "ly",     # thinly, lightly, roughly
    "ed",     # toasted, chilled, beaten
)


def normalize_ascii(word):
    return unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii")

def filter_base_words(words, stop_words = BAD_WORDS):
    stop_words = {w.lower() for w in stop_words}

    cleaned = []

    for w in words:
        ascii_w = normalize_ascii(w)

        if (
            ascii_w.isalpha()
            and len(ascii_w) >= 3
            and ascii_w.lower() not in stop_words
        ):
            cleaned.append(ascii_w)

    # Sort by length
    cleaned = sorted(cleaned, key=len)

    result = []

    while cleaned:
        base = cleaned[0]
        result.append(base)

        cleaned = [
            w for w in cleaned
            if w == base or base.lower() not in w.lower()
        ]

        cleaned.remove(base)

    return result

def clean_and_extract(ingredients_str):

    # Convert string to list
    arr = ast.literal_eval(ingredients_str)
    
    cleaned = []
    for item in arr:
        # Remove parentheses
        item = re.sub(r'\([^)]*\)', '', item)
        # Lowercase
        item = item.lower()
        # Remove numbers, fractions, ranges
        item = re.sub(r'\b\d+([\/.-]\d+)?\b', '', item)
        # Remove punctuation
        item = re.sub(r'[:*,;-]', '', item)
        # Replace stray hyphens with space
        item = re.sub(r'-+', ' ', item)
        # Normalize spaces
        item = re.sub(r'\s+', ' ', item)
        # Trim
        item = item.strip()
        # Split, remove stop words, join back
        words = [word for word in item.split(' ') if word and word not in BAD_WORDS]
        cleaned_item = ' '.join(words)
        if cleaned_item:
            cleaned.append(cleaned_item)
    

    return cleaned

def to_singular(word: str) -> str:
    singular = p.singular_noun(word)
    return singular if singular else word



def main():

    df = pd.read_csv(CSV_PATH)

    counter = Counter()

    all_tokens = []

    # Build frequency of clean ingredient words
    for ingredients in df["Cleaned_Ingredients"]:
        tokens = clean_and_extract(ingredients)
        base_tokens = filter_base_words(tokens)
        singular_tokens = [to_singular(token) for token in base_tokens]
        all_tokens.extend(singular_tokens)
        counter.update(singular_tokens)

    filtered = []

    for word, count in counter.most_common():
        if word in BAD_WORDS:
            continue
        if word.endswith(BAD_SUFFIXES):
            continue
        if count < MIN_FREQ:
            continue
        filtered.append((word, count))

    most_common = filtered[:TOP_K]


    # most_common = counter.most_common(TOP_K)

    vocab = {
        ingredient: idx
        for idx, (ingredient, _) in enumerate(most_common)
    }

    print(f"Final vocabulary size: {len(vocab)}")

    with open(OUT_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    print("Saved vocab to:", OUT_PATH)


if __name__ == "__main__":
    main()
