import re
import html

def clean_review(text):
    #fix HTML escapes
    text = html.unescape(text)

    #lowercase
    text = text.lower()

    #remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    #remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    #remove user mentions
    text = re.sub(r'@\w+', '', text)

    #remove hashtags (keep text)
    text = re.sub(r'#(\w+)', r'\1', text)

    #expand contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'t": " not", "'ve": " have", "'m": " am"
    }
    for c, e in contractions.items():
        text = text.replace(c, e)

    #remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    #remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

sample = "This product is AMAZING!!! Worth every ₹, will buy again 😍😍 https://amazon.com"
print(clean_review(sample))
