import nltk

# Download basic required packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw')

# Special requirement for text2emotion
nltk.download('punkt_tab')  # This is what's missing

print("NLTK data download complete!")

import text2emotion as te
print(te.get_emotion("I am feeling very sad today"))