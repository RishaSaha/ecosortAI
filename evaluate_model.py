import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. TRAINING DATA
# ============================================================

training_data = [

    # --------------------------------------------------------
    # RECYCLABLE
    # --------------------------------------------------------
    ("plastic water bottle", "Recyclable"),
    ("empty plastic bottle", "Recyclable"),
    ("plastic soda bottle", "Recyclable"),
    ("clean plastic container", "Recyclable"),
    ("plastic food container", "Recyclable"),
    ("plastic packaging", "Recyclable"),
    ("metal soda can", "Recyclable"),
    ("aluminium can", "Recyclable"),
    ("empty tin can", "Recyclable"),
    ("steel food can", "Recyclable"),
    ("cardboard box", "Recyclable"),
    ("shipping cardboard box", "Recyclable"),
    ("clean cardboard packaging", "Recyclable"),

    # Added paper-related examples
    ("newspaper", "Recyclable"),
    ("daily newspaper", "Recyclable"),
    ("printed newspaper", "Recyclable"),
    ("paper magazine", "Recyclable"),
    ("monthly magazine", "Recyclable"),
    ("printed magazine", "Recyclable"),
    ("glossy magazine", "Recyclable"),
    ("old book", "Recyclable"),
    ("paper book", "Recyclable"),
    ("printed book", "Recyclable"),
    ("catalog", "Recyclable"),
    ("printed catalog", "Recyclable"),
    ("paper brochure", "Recyclable"),
    ("paper leaflet", "Recyclable"),

    ("glass bottle", "Recyclable"),
    ("glass jar", "Recyclable"),
    ("empty glass container", "Recyclable"),

    # --------------------------------------------------------
    # ORGANIC
    # --------------------------------------------------------
    ("banana peel", "Organic"),
    ("banana peels", "Organic"),
    ("apple peel", "Organic"),
    ("fruit peel", "Organic"),
    ("fruit scraps", "Organic"),
    ("vegetable scraps", "Organic"),
    ("vegetable peels", "Organic"),
    ("kitchen food waste", "Organic"),
    ("leftover food", "Organic"),
    ("leftover rice", "Organic"),
    ("food leftovers", "Organic"),
    ("rotten fruit", "Organic"),
    ("rotten vegetables", "Organic"),
    ("coffee grounds", "Organic"),
    ("tea leaves", "Organic"),
    ("garden leaves", "Organic"),
    ("fallen leaves", "Organic"),
    ("grass clippings", "Organic"),
    ("compostable food scraps", "Organic"),

    # --------------------------------------------------------
    # E-WASTE
    # --------------------------------------------------------
    ("broken laptop", "E-waste"),
    ("old laptop", "E-waste"),
    ("damaged computer", "E-waste"),
    ("desktop computer", "E-waste"),
    ("old smartphone", "E-waste"),
    ("broken phone", "E-waste"),
    ("mobile phone", "E-waste"),
    ("old tablet", "E-waste"),
    ("computer mouse", "E-waste"),
    ("computer keyboard", "E-waste"),
    ("broken monitor", "E-waste"),
    ("computer monitor", "E-waste"),
    ("television", "E-waste"),
    ("broken television", "E-waste"),
    ("phone charger", "E-waste"),
    ("damaged charger", "E-waste"),
    ("USB cable", "E-waste"),

    # Added audio/electronic accessories
    ("wireless headphones", "E-waste"),
    ("bluetooth headphones", "E-waste"),
    ("wired headphones", "E-waste"),
    ("earphones", "E-waste"),
    ("wireless earbuds", "E-waste"),
    ("bluetooth earbuds", "E-waste"),
    ("electronic earbuds", "E-waste"),
    ("portable speaker", "E-waste"),
    ("bluetooth speaker", "E-waste"),
    ("electronic speaker", "E-waste"),
    ("electronic accessories", "E-waste"),

    ("digital camera", "E-waste"),
    ("old camera", "E-waste"),
    ("smartwatch", "E-waste"),
    ("electronic device", "E-waste"),

    # --------------------------------------------------------
    # HAZARDOUS
    # --------------------------------------------------------
    ("used battery", "Hazardous"),
    ("dead battery", "Hazardous"),
    ("lithium ion battery", "Hazardous"),
    ("lithium battery", "Hazardous"),
    ("old battery cell", "Hazardous"),
    ("button cell battery", "Hazardous"),
    ("alkaline battery", "Hazardous"),

    ("expired medicine", "Hazardous"),
    ("expired tablets", "Hazardous"),
    ("old medicine", "Hazardous"),
    ("unused medication", "Hazardous"),
    ("medicine waste", "Hazardous"),

    ("used syringe", "Hazardous"),
    ("medical needle", "Hazardous"),
    ("used injection needle", "Hazardous"),
    ("sharp medical waste", "Hazardous"),

    # Added chemical/agricultural waste
    ("pesticide bottle", "Hazardous"),
    ("pesticide waste", "Hazardous"),
    ("pesticide container", "Hazardous"),
    ("insecticide container", "Hazardous"),
    ("insecticide waste", "Hazardous"),
    ("herbicide container", "Hazardous"),
    ("agricultural chemical container", "Hazardous"),
    ("chemical container", "Hazardous"),
    ("household chemical waste", "Hazardous"),
    ("toxic chemical waste", "Hazardous"),
    ("paint thinner", "Hazardous"),
    ("chemical solvent", "Hazardous"),

    # --------------------------------------------------------
    # RESIDUAL
    # --------------------------------------------------------
    ("used tissue", "Residual"),
    ("facial tissue", "Residual"),
    ("dirty tissue", "Residual"),
    ("used napkin", "Residual"),
    ("dirty napkin", "Residual"),
    ("paper towel", "Residual"),
    ("used paper towel", "Residual"),
    ("dirty paper towel", "Residual"),
    ("baby diaper", "Residual"),
    ("used diaper", "Residual"),
    ("sanitary waste", "Residual"),
    ("broken ceramic", "Residual"),
    ("broken ceramic plate", "Residual"),
    ("ceramic cup", "Residual"),
    ("chips wrapper", "Residual"),
    ("snack wrapper", "Residual"),
    ("food wrapper", "Residual"),
    ("dirty plastic wrapper", "Residual"),
]


# Convert training data into DataFrame
train_df = pd.DataFrame(
    training_data,
    columns=["text", "category"]
)


# ============================================================
# 2. MODEL
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# Train the model
model.fit(
    train_df["text"],
    train_df["category"]
)


# ============================================================
# 3. HELD-OUT TEST SET
# ============================================================
# IMPORTANT:
# These examples are kept unchanged from our original test.
# DO NOT modify them when comparing model versions.
# ============================================================

test_data = [

    # Recyclable
    ("empty Pepsi bottle", "Recyclable"),
    ("clean mineral water bottle", "Recyclable"),
    ("metal soda can", "Recyclable"),
    ("shipping box", "Recyclable"),
    ("old magazine", "Recyclable"),
    ("glass food jar", "Recyclable"),

    # Organic
    ("rotten banana and apple peels", "Organic"),
    ("leftover rice from dinner", "Organic"),
    ("kitchen vegetable scraps", "Organic"),
    ("fallen garden leaves", "Organic"),
    ("used coffee grounds", "Organic"),
    ("fruit leftovers", "Organic"),

    # E-waste
    ("my broken laptop", "E-waste"),
    ("discarded smartphone", "E-waste"),
    ("old computer mouse", "E-waste"),
    ("damaged phone charger", "E-waste"),
    ("unused headphones", "E-waste"),
    ("broken television", "E-waste"),

    # Hazardous
    ("dead batteries from my remote", "Hazardous"),
    ("old lithium ion cell", "Hazardous"),
    ("expired tablets", "Hazardous"),
    ("used injection needle", "Hazardous"),
    ("old pesticide container", "Hazardous"),
    ("household chemical waste", "Hazardous"),

    # Residual
    ("used facial tissue", "Residual"),
    ("dirty napkin", "Residual"),
    ("old paper towel", "Residual"),
    ("used baby diaper", "Residual"),
    ("broken ceramic plate", "Residual"),
    ("empty chips wrapper", "Residual"),
]


test_df = pd.DataFrame(
    test_data,
    columns=["text", "category"]
)


# ============================================================
# 4. PREDICTIONS
# ============================================================

predictions = model.predict(test_df["text"])


# ============================================================
# 5. ACCURACY
# ============================================================

accuracy = accuracy_score(
    test_df["category"],
    predictions
)

print("=" * 60)
print("ECOSORT AI - MODEL EVALUATION")
print("=" * 60)

print(f"\nTest samples: {len(test_df)}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        test_df["category"],
        predictions
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

labels = [
    "Recyclable",
    "Organic",
    "E-waste",
    "Hazardous",
    "Residual"
]

cm = confusion_matrix(
    test_df["category"],
    predictions,
    labels=labels
)

print("Confusion Matrix:")
print("Labels:", labels)
print(cm)


# ============================================================
# 8. SHOW INDIVIDUAL RESULTS
# ============================================================

print("\nIndividual Test Results:")
print("-" * 60)

for text, actual, predicted in zip(
    test_df["text"],
    test_df["category"],
    predictions
):

    status = "✓" if actual == predicted else "✗"

    print(
        f"{status} {text}"
        f" | Actual: {actual}"
        f" | Predicted: {predicted}"
    )


# ============================================================
# 9. SHOW ONLY MISTAKES
# ============================================================

print("\nMistakes:")
print("-" * 60)

mistake_count = 0

for text, actual, predicted in zip(
    test_df["text"],
    test_df["category"],
    predictions
):

    if actual != predicted:

        mistake_count += 1

        print(
            f"{mistake_count}. {text}"
            f" | Actual: {actual}"
            f" | Predicted: {predicted}"
        )


if mistake_count == 0:
    print("No mistakes on the held-out test set.")

print("\n" + "=" * 60)