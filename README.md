
# 🗑️ EcoSort AI

### An AI-Powered Waste Segregation and Sustainable Disposal Assistant

EcoSort AI is a Streamlit-based prototype that helps users identify likely waste categories and understand appropriate disposal approaches through natural-language waste descriptions.

The project combines machine learning, safety-focused detection rules, and item-specific knowledge retrieval to provide more informative waste-management guidance.

---

## 🌱 Sustainable Development Goal

**Primary SDG:** SDG 12 — Responsible Consumption and Production

**Related SDGs:**
- SDG 11 — Sustainable Cities and Communities
- SDG 13 — Climate Action

---

## 🎯 Problem Statement

People often find it difficult to identify the correct category of everyday waste and understand how it should be disposed of.

Incorrect waste segregation can contaminate recyclable materials and create risks when hazardous or special waste is mixed with ordinary household waste.

EcoSort AI aims to provide an accessible prototype that helps users understand waste categories and explore safer disposal approaches.

---

## 💡 Proposed Solution

Users describe a waste item in natural language, such as:

> I have an old laptop that no longer works.

The application provides:

- A predicted waste category
- A model confidence estimate
- Category-specific information
- Item-specific disposal guidance when available
- Sustainability tips
- Safety warnings for potentially hazardous waste
- A reminder to verify local waste-management rules

---

## 🤖 AI and Technical Components

- **Python** — Main programming language
- **Streamlit** — Interactive web application
- **TF-IDF** — Text feature representation
- **Logistic Regression** — Waste-category classification
- **Keyword-based retrieval** — Item-specific guidance
- **Rule-based safety detection** — Identification of potentially hazardous waste
- **Structured knowledge base** — Disposal guidance and sustainability information

---

## ♻️ Waste Categories

The prototype currently supports the following categories:

1. Recyclable
2. Organic
3. E-waste
4. Hazardous
5. Residual

---

## 📊 Model Evaluation

The improved TF-IDF and Logistic Regression prototype achieved:

- **Accuracy:** 100.00%
- **Macro F1-score:** 1.00
- **Evaluation samples:** 30 curated test descriptions
- **Categories evaluated:** 5

The initial model achieved 90.00% accuracy on the same held-out test set. Error analysis was used to improve the training examples before re-evaluation.

> Note: The evaluation uses a small, curated dataset and should not be interpreted as real-world accuracy. Performance may differ for unfamiliar waste descriptions.

---

## 🛡️ Responsible AI

EcoSort AI is an educational prototype and its predictions are not guaranteed to be correct for every waste item.

The application includes:
- Confidence-related warnings
- Safety detection rules for selected hazardous items
- Guidance to verify local disposal requirements
- A reminder that waste-management rules vary by location

The application does not replace official guidance from local authorities or authorized waste-management facilities.

Users should take additional care when handling hazardous, medical, chemical, battery, or electronic waste.

---

## 🚀 Running the Application Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ecosortAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser through a local Streamlit address.

---

## 📁 Project Structure

```text
ecosortAI/
│
├── app.py
├── evaluate_model.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

Possible future improvements include:

- Expanding the training dataset
- Adding image-based waste recognition
- Improving classification for unfamiliar descriptions
- Adding location-specific disposal information
- Introducing a more extensive waste-management knowledge base
- Conducting evaluation on a larger and more diverse dataset

---

## ⚠️ Project Disclaimer

EcoSort AI is a prototype developed for educational and sustainability-focused purposes.

Always verify disposal instructions with local authorities or authorized waste facilities, particularly for hazardous and special waste.