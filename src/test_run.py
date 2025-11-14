
import os
print("Running file:", os.path.abspath(__file__))
from data_utils import load_reviews, clean_text, shuffle_data, split_data

# 1. Load reviews
data = load_reviews(
    r"C:\Users\Tani Santi Dewi Jaya\nlp-sentiment-amazon\data\positive.review",
    r"C:\Users\Tani Santi Dewi Jaya\nlp-sentiment-amazon\data\negative.review"
)

print("Loaded reviews:", len(data))
print("Example raw review:", data[0][0][:200])

# 2. Clean reviews
cleaned_data = [(clean_text(review), label) for review, label in data]

for review, label in data:
    if len(review.strip()) > 2 and "<review>" not in review:
        cleaned = clean_text(review)
        print("Raw review (real):", review[:200])
        print("Cleaned review (real):", cleaned[:200])
        break

# 3. Shuffle
cleaned_data = shuffle_data(cleaned_data)

# 4. Split into train/test
X_train, X_test, y_train, y_test = split_data(cleaned_data, test_size=0.2)

print("Sample train text:", X_train[0][:200])
print("Train size:", len(X_train))
print("Test size:", len(X_test))
