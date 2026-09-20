import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered"
)

# ---------------------------------------------------------
# CUSTOM UI STYLING
# ---------------------------------------------------------

st.markdown("""
<style>

    /* -------------------------------------------------
       MAIN APPLICATION BACKGROUND
    ------------------------------------------------- */

    .stApp {
        background-color: #f4f8f4;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f4f8f4;
    }

    /* Main content text */
    [data-testid="stAppViewContainer"] .main {
        color: #1f2937;
    }


    /* -------------------------------------------------
       MAIN TITLE
    ------------------------------------------------- */

    h1 {
        color: #1b5e20 !important;
        font-weight: 700;
    }


    /* -------------------------------------------------
       SECTION HEADINGS
    ------------------------------------------------- */

    h2,
    h3 {
        color: #2e7d32 !important;
    }


    /* -------------------------------------------------
       GENERAL TEXT VISIBILITY
    ------------------------------------------------- */

    .main p,
    .main li,
    .main span,
    .main label {
        color: #1f2937;
    }

    /* Markdown text */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #1f2937;
    }


    /* -------------------------------------------------
       TEXT INPUT AND TEXT AREA
    ------------------------------------------------- */

    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        padding: 12px;
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #a5d6a7;
        padding: 12px;
        color: #1f2937 !important;
        background-color: #ffffff !important;
    }

    /* Text input and text area placeholder */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #6b7280 !important;
        opacity: 1 !important;
    }

    /* Widget labels */
    [data-testid="stWidgetLabel"] p {
        color: #1f2937 !important;
    }


    /* -------------------------------------------------
       BUTTONS
    ------------------------------------------------- */

    .stButton > button {
        background-color: #2e7d32;
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 10px 22px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1b5e20;
        color: white !important;
    }


    /* -------------------------------------------------
       INFORMATION BOXES
    ------------------------------------------------- */

    .stAlert {
        border-radius: 10px;
    }


    /* -------------------------------------------------
       EXPANDERS
    ------------------------------------------------- */

    [data-testid="stExpander"] {
        color: #1f2937;
    }

    [data-testid="stExpander"] p {
        color: #1f2937;
    }


    /* -------------------------------------------------
       SIDEBAR
    ------------------------------------------------- */

    [data-testid="stSidebar"] {
        background-color: #262730;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span {
        color: #ffffff;
    }

    /* -------------------------------------------------
   DISCLAIMER AND SMALL TEXT VISIBILITY
------------------------------------------------- */

/* Streamlit caption text */
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #374151 !important;
    opacity: 1 !important;
}

/* Markdown disclaimer text */
.main small,
.main small p {
    color: #374151 !important;
    opacity: 1 !important;
}

/* All normal markdown text */
[data-testid="stMarkdownContainer"] p {
    color: #1f2937 !important;
}

/* Disclaimer inside custom HTML containers */
.disclaimer,
.disclaimer p {
    color: #374151 !important;
    background-color: #e8f5e9;
    padding: 12px;
    border-radius: 8px;
    opacity: 1 !important;
}

/* -------------------------------------------------
   FINAL SIDEBAR TEXT OVERRIDE
   Keep sidebar content readable on dark background
------------------------------------------------- */

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #262730 !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}

/* Sidebar normal text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

/* Sidebar markdown text */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
    color: #ffffff !important;
}

/* Sidebar caption / Important Notice */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #ffffff !important;
    opacity: 1 !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #555555 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR: PROJECT INFORMATION
# ---------------------------------------------------------

with st.sidebar:

    st.header("🌱 About EcoSort AI")

    st.write(
        "EcoSort AI is a prototype assistant that helps users "
        "understand waste categories and explore appropriate "
        "disposal methods."
    )

    st.divider()

    st.subheader("🎯 Sustainable Development Goal")

    st.write(
        "SDG 12: Responsible Consumption and Production"
    )

    st.divider()

    st.subheader("🤖 AI Used")

    st.write(
        "• TF-IDF text representation\n"
        "• Logistic Regression classification\n"
        "• Keyword-based knowledge retrieval\n"
        "• Safety detection rules"
    )

    st.divider()

    st.subheader("⚠️ Important Notice")

    st.caption(
        "This is an experimental prototype trained on a "
        "curated dataset. Predictions may be incorrect. "
        "Verify local disposal rules, particularly for "
        "hazardous and special waste."
    )


# ============================================================
# TITLE
# ============================================================

st.title("🗑️ EcoSort AI")

st.markdown(
    "### Your AI-powered assistant for smarter waste segregation"
)

st.write(
    "Describe an everyday waste item to receive a likely waste "
    "category, disposal guidance, and a sustainability tip."
)

st.info(
    "♻️ Prototype notice: Waste-management rules vary by location. "
    "Always verify disposal instructions with your local authority, "
    "especially for hazardous waste."
)

# ============================================================
# KNOWLEDGE BASE
# ============================================================

waste_info = {

    "Recyclable": {
        "description": (
            "Materials that may be recoverable through recycling "
            "when they are clean and accepted by the local recycling system."
        ),
        "guidance": (
            "Keep the item clean and dry where appropriate and place it "
            "in the recycling stream only if your local waste system accepts it."
        ),
        "tip": (
            "Rinse reusable containers when practical and avoid contaminating "
            "other recyclable materials with food or liquid."
        )
    },

    "Organic": {
        "description": (
            "Biodegradable food and plant waste that can potentially be "
            "composted or processed as organic waste."
        ),
        "guidance": (
            "Place food scraps or plant waste in the organic/composting stream "
            "if this service is available in your area."
        ),
        "tip": (
            "Composting suitable organic waste can reduce the amount of "
            "biodegradable material sent to landfill."
        )
    },

    "E-waste": {
        "description": (
            "Discarded electrical or electronic equipment and accessories "
            "that require specialized collection or recycling."
        ),
        "guidance": (
            "Do not place electronic devices in ordinary household waste. "
            "Use an authorized e-waste collection or recycling facility."
        ),
        "tip": (
            "Repair, reuse, donate, or responsibly recycle electronics "
            "before replacing them whenever practical."
        )
    },

    "Hazardous": {
        "description": (
            "Waste that may contain hazardous substances or present "
            "chemical, medical, fire, or environmental risks."
        ),
        "guidance": (
            "Keep hazardous waste separate from ordinary household waste "
            "and follow your local hazardous-waste collection guidance."
        ),
        "tip": (
            "Never mix unknown chemicals or dispose of potentially hazardous "
            "materials through ordinary recycling."
        )
    },

    "Residual": {
        "description": (
            "Waste that generally cannot be recycled or composted through "
            "the systems represented by this prototype."
        ),
        "guidance": (
            "Place it in the residual/general-waste stream according to "
            "your local waste-management rules."
        ),
        "tip": (
            "Before throwing something away, check whether it can be "
            "reused, repaired, recycled, or composted."
        )
    }
}

# ---------------------------------------------------------
# ITEM-SPECIFIC WASTE KNOWLEDGE BASE
# Used later for retrieval-grounded guidance
# ---------------------------------------------------------

waste_knowledge = {
    "plastic bottle": {
        "category": "Recyclable",
        "guidance": "Empty and rinse the bottle when practical. Keep it clean and dry and check whether your local recycling system accepts plastic bottles.",
        "tip": "Reuse refillable bottles when possible to reduce single-use plastic waste."
    },

    "plastic container": {
        "category": "Recyclable",
        "guidance": "Empty and clean the container when practical. Place it in recycling only if the material is accepted locally.",
        "tip": "Choose reusable containers where possible."
    },

    "metal can": {
        "category": "Recyclable",
        "guidance": "Empty and rinse the can when practical before placing it in the recycling stream, if accepted locally.",
        "tip": "Recycling metal can save raw materials and energy compared with producing new metal."
    },

    "cardboard": {
        "category": "Recyclable",
        "guidance": "Keep cardboard reasonably clean and dry. Flatten large boxes before recycling where appropriate.",
        "tip": "Reuse shipping boxes before recycling them."
    },

    "paper": {
        "category": "Recyclable",
        "guidance": "Keep paper clean and dry and place it in the paper/recycling stream if accepted locally.",
        "tip": "Use both sides of paper before recycling when practical."
    },

    "glass bottle": {
        "category": "Recyclable",
        "guidance": "Empty and rinse the bottle and check whether your local recycling system accepts glass.",
        "tip": "Reuse glass containers when practical before recycling them."
    },

    "banana peel": {
        "category": "Organic",
        "guidance": "Place the peel in an organic-waste or composting stream if available in your area.",
        "tip": "Composting suitable food scraps can reduce biodegradable waste sent to landfill."
    },

    "vegetable scraps": {
        "category": "Organic",
        "guidance": "Place suitable vegetable scraps in an organic-waste or composting system where available.",
        "tip": "Collecting food scraps for composting can return nutrients to soil."
    },

    "food scraps": {
        "category": "Organic",
        "guidance": "Place suitable food waste in the organic/composting stream if your local service accepts it.",
        "tip": "Reducing food waste at the source is preferable to disposing of edible food."
    },

    "coffee grounds": {
        "category": "Organic",
        "guidance": "Coffee grounds can generally be directed to a suitable composting or organic-waste system where available.",
        "tip": "Composting coffee grounds can help keep organic material out of general waste."
    },

    "mobile phone": {
        "category": "E-waste",
        "guidance": "Do not place the phone in ordinary household waste. Use an authorized e-waste collection or recycling facility.",
        "tip": "Repair, reuse, donate, or responsibly recycle electronics before replacing them when practical."
    },

    "laptop": {
        "category": "E-waste",
        "guidance": "Do not place the laptop in ordinary household waste. Use an authorized e-waste collection or recycling facility.",
        "tip": "Consider repair, reuse, refurbishment, or donation before recycling."
    },

    "charger": {
        "category": "E-waste",
        "guidance": "Electronic chargers should be kept separate from ordinary household waste and taken to an appropriate e-waste collection point.",
        "tip": "Reuse compatible chargers when safe and practical instead of replacing them unnecessarily."
    },

    "headphones": {
        "category": "E-waste",
        "guidance": "Do not place electronic headphones in ordinary household waste. Use an appropriate e-waste collection or recycling facility.",
        "tip": "Repair or reuse working electronic accessories before replacing them."
    },

    "battery": {
        "category": "Hazardous",
        "guidance": "Keep batteries separate from ordinary household waste and follow local battery or hazardous-waste collection guidance.",
        "tip": "Never puncture, crush, burn, or place batteries in ordinary recycling."
    },

    "lithium battery": {
        "category": "Hazardous",
        "guidance": "Keep lithium batteries separate from ordinary household waste and use an authorized battery or hazardous-waste collection service.",
        "tip": "Damaged lithium batteries require extra care because they can present fire and chemical hazards."
    },

    "medicine": {
        "category": "Hazardous",
        "guidance": "Do not place medicines in ordinary household waste unless local guidance specifically permits it. Use an authorized medicine take-back or disposal service where available.",
        "tip": "Avoid flushing medicines unless official local guidance specifically instructs you to do so."
    },

    "syringe": {
        "category": "Hazardous",
        "guidance": "Do not place used syringes or needles in ordinary household waste or recycling. Use an authorized sharps-disposal service.",
        "tip": "Never handle or recycle loose needles with bare hands."
    },

    "pesticide container": {
        "category": "Hazardous",
        "guidance": "Keep pesticide containers separate from ordinary household waste and follow local hazardous-waste or agricultural disposal guidance.",
        "tip": "Do not reuse containers that held pesticides or other hazardous chemicals."
    },

    "tissue": {
        "category": "Residual",
        "guidance": "Used tissues are generally not suitable for recycling. Dispose of them through the residual/general-waste stream according to local rules.",
        "tip": "Use only the amount of tissue needed and choose reusable alternatives where practical."
    },

    "diaper": {
        "category": "Residual",
        "guidance": "Used diapers are generally treated as residual/general waste. Follow your local household-waste rules.",
        "tip": "Where practical, consider reusable alternatives to reduce single-use waste."
    },

    "ceramic": {
        "category": "Residual",
        "guidance": "Broken ceramic items are generally not accepted in standard recycling streams. Follow local disposal guidance.",
        "tip": "Reuse intact ceramic items where possible before disposal."
    },

    "food wrapper": {
        "category": "Residual",
        "guidance": "Many food wrappers are not accepted in standard recycling systems. Check local guidance before placing them in recycling.",
        "tip": "Choose products with reusable or easily recyclable packaging where practical."
    }
}


# ============================================================
# TRAINING DATA
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

    # Paper-related examples
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

    # Audio/electronic accessories
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

    # Chemical/agricultural waste
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

# ---------------------------------------------------------
# KNOWLEDGE RETRIEVAL
# Finds the most specific matching item guidance
# ---------------------------------------------------------

def retrieve_waste_knowledge(text):

    text = text.lower()

    # Check longer keywords first.
    # This ensures specific phrases are matched before
    # shorter, more general keywords.
    sorted_keywords = sorted(
        waste_knowledge.keys(),
        key=len,
        reverse=True
    )

    for keyword in sorted_keywords:

        if keyword in text:

            return waste_knowledge[keyword]

    # No specific item found
    return None


# ============================================================
# TRAIN MODEL
# ============================================================

texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]


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


model.fit(texts, labels)


# ============================================================
# SPECIAL WASTE SAFETY DETECTION
# ============================================================

def detect_special_waste(text):

    text_lower = text.lower()

    # --------------------------------------------------------
    # Battery safety
    # --------------------------------------------------------

    battery_words = [
        "battery",
        "lithium",
        "lithium-ion",
        "button cell",
        "power bank"
    ]

    if any(word in text_lower for word in battery_words):

        dangerous_battery_words = [
            "swollen",
            "damaged",
            "leaking",
            "leaked",
            "punctured",
            "hot",
            "burning"
        ]

        if any(
            word in text_lower
            for word in dangerous_battery_words
        ):

            return (
                "Hazardous",
                0.98,
                "⚠️ Potentially damaged battery detected. "
                "Do not puncture, crush, burn, or place it in "
                "ordinary household waste. Follow local hazardous "
                "battery disposal guidance."
            )

        return (
            "Hazardous",
            0.97,
            "⚠️ Batteries should be kept separate from ordinary "
            "household waste and taken to an appropriate battery "
            "collection or hazardous-waste facility."
        )


    # --------------------------------------------------------
    # Medicine
    # --------------------------------------------------------

    medicine_words = [
        "medicine",
        "medication",
        "tablet",
        "tablets",
        "capsule",
        "pills"
    ]

    if any(word in text_lower for word in medicine_words):

        return (
            "Hazardous",
            0.96,
            "⚠️ Medication should not automatically be placed "
            "in ordinary household waste. Follow local pharmaceutical "
            "take-back or disposal guidance."
        )


    # --------------------------------------------------------
    # Sharps / medical waste
    # --------------------------------------------------------

    sharp_words = [
        "syringe",
        "needle",
        "injection",
        "sharp medical"
    ]

    if any(word in text_lower for word in sharp_words):

        return (
            "Hazardous",
            0.99,
            "⚠️ Sharps can cause injury and must not be placed "
            "loose in ordinary household waste. Use an appropriate "
            "medical or sharps disposal service."
        )


    # --------------------------------------------------------
    # Chemical / pesticide waste
    # --------------------------------------------------------

    chemical_words = [
        "pesticide",
        "insecticide",
        "herbicide",
        "chemical",
        "solvent",
        "paint thinner",
        "toxic chemical"
    ]

    if any(word in text_lower for word in chemical_words):

        return (
            "Hazardous",
            0.96,
            "⚠️ Potentially hazardous chemical waste detected. "
            "Keep it separate from ordinary waste and follow "
            "local hazardous-waste disposal guidance."
        )


    return None


# ============================================================
# USER INPUT
# ============================================================

user_input = st.text_area(
    "🗑️ Describe your waste item",
    placeholder=(
        "Example: I have an old pair of Bluetooth headphones "
        "that no longer works."
    ),
    height=120
)

# ============================================================
# EXAMPLE INPUTS
# ============================================================

with st.expander("💡 Need an example? Try one of these"):

    st.write("You can describe waste items such as:")

    st.markdown("""
    - 🥤 Empty plastic water bottle
    - 🍌 Banana peels from breakfast
    - 💻 Old laptop that no longer works
    - 🔋 Damaged lithium battery
    - 🧴 Empty pesticide container
    - 🧻 Used paper tissue
    """)

    st.caption(
        "Copy an example into the input box and click "
        "'Analyze Waste' to test the application."
    )


# ============================================================
# CLASSIFY
# ============================================================

if st.button("🔍 Analyze Waste", use_container_width=True):

    if not user_input.strip():

        st.warning("Please describe a waste item first.")

    else:

        # Always initialize category and confidence.
        # This prevents the NameError that occurred previously.
        category = None
        confidence = 0.0

        special_result = detect_special_waste(user_input)


        # ----------------------------------------------------
        # SAFETY LAYER RESULT
        # ----------------------------------------------------

        if special_result is not None:

            category, confidence, warning = special_result

            # Display the safety warning.
            # Category and confidence will be displayed later
            # in the structured Waste Assessment section.
            st.warning(warning)


        # ----------------------------------------------------
        # NORMAL ML CLASSIFICATION
        # ----------------------------------------------------

        else:

            probabilities = model.predict_proba(
                [user_input]
            )[0]

            category = model.predict(
                [user_input]
            )[0]

            confidence = max(probabilities)


            # Confidence explanation

            if confidence < 0.55:

                st.warning(
                    "⚠️ The model has relatively low confidence. "
                    "Consider checking the item against your local "
                    "waste-management guidance."
                )

            elif confidence < 0.75:

                st.info(
                    "ℹ️ The model has moderate confidence. "
                    "Local disposal rules should still be checked."
                )

            else:

                st.info(
                    "The model has relatively high confidence for "
                    "this prototype classification."
                )



        # ----------------------------------------------------
        # RETRIEVE GUIDANCE
        # ----------------------------------------------------

        specific_info = retrieve_waste_knowledge(user_input)

        # Use item-specific guidance when available.
        # Otherwise, use general category guidance.
        if specific_info is not None:

            info = specific_info

        else:

            info = waste_info[category]


        # ----------------------------------------------------
        # STRUCTURED RESULT DISPLAY
        # ----------------------------------------------------

        st.divider()

        st.subheader("📋 Waste Assessment")

        # Display category and confidence in two columns
        col1, col2 = st.columns(2)

        with col1:
         st.metric(
          label="Waste Category",
          value=category
        )

        with col2:
            st.metric(
                label="Confidence Estimate",
                value="Rule-based" if special_result is not None else f"{confidence * 100:.1f}%"
            )

        st.divider()

        # Category meaning
        with st.container(border=True):

         st.subheader("📌 What does this category mean?")

         st.write(
          waste_info[category]["description"]
        )

       # Disposal guidance
        with st.container(border=True):

         st.subheader("♻️ Suggested disposal approach")

         st.write(
          info["guidance"]
        )

        # Sustainability tip
        with st.container(border=True):

         st.subheader("🌱 Sustainability tip")

         st.write(
           info["tip"]
        )

# ============================================================
# RESPONSIBLE AI NOTICE
# ============================================================

st.divider()

st.markdown(
    "### 🛡️ Responsible AI Notice"
)

st.caption(
    "EcoSort AI is an educational prototype trained on a curated "
    "dataset. Its predictions are not guaranteed to be correct for "
    "every waste item. Waste-management rules vary by location, "
    "so users should verify disposal instructions with local "
    "authorities or authorized waste facilities, especially for "
    "hazardous, medical, chemical, or electronic waste."
)