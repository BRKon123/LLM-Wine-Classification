1. **Create a Virtual Environment**

   First, ensure you have Python installed on your system. You can create a virtual environment by running the following command in your terminal or command prompt:

   ```bash
   python -m venv venv
   ```

   This will create a directory named `venv` in your project folder.

2. **Activate the Virtual Environment**

   - On **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - On **macOS and Linux**:

     ```bash
     source venv/bin/activate
     ```

   Once activated, your terminal prompt should change to indicate that you are now working inside the virtual environment.

3. **Install Dependencies**

   With the virtual environment activated, install the required packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   This command will read the `requirements.txt` file and install all the listed packages.

4. **Deactivate the Virtual Environment**

   When you're done working in the virtual environment, you can deactivate it by simply running:

   ```bash
   deactivate
   ```

### Code

1. **Wine Ratings Visualization** (`src/wine_ratings_visualisation.ipynb`)

   - Purpose: Exploratory Data Analysis (EDA) of wine dataset
     - Visualizes distribution of wine ratings and prices
     - Creates country-wise comparisons
     - Generates plots for wine varieties
     - Provides insights into wine description patterns
   - Dependencies: pandas, matplotlib, seaborn

2. **Traditional NLP Analysis** (`src/wine_ratings_traditional_nlp.ipynb`)

   - Purpose: Machine Learning and NLP analysis of wine descriptions
     - Text preprocessing and vectorization
     - Feature engineering (TF-IDF, sentiment analysis)
     - Implements multiple ML models:
       - Logistic Regression
       - Random Forest Classifier

3. **Basic LLM Classification** (`src/llm.py`)

   - Purpose: Direct wine classification using LLM
   - Features:
     - Uses OpenAI's API for wine country prediction
     - Processes wine descriptions and metadata
     - Supports Chain-of-Thought (CoT) reasoning
     - Tracks and reports prediction accuracy
   - Implementation:
     - Tests on last 200 entries of dataset
     - Formats wine information for LLM consumption
     - Extracts predictions from LLM responses
     - Handles API errors gracefully

4. **RAG-Enhanced Classification** (`src/rag_llm.py`)
   - Purpose: Enhanced classification using Retrieval-Augmented Generation (RAG)
   - Features:
     - Combines LLM with similar wine examples
     - Uses ChromaDB for similarity search
     - Embeds training data (first 800 entries)
     - Retrieves relevant examples for each prediction
   - Implementation:
     - Splits data into training (800) and testing (200)
     - Embeds wine descriptions using ChromaDB
     - Retrieves 3 similar wines for context
     - Includes similar wines in LLM prompt
     - Supports Chain-of-Thought reasoning

#### Traditional ML Notes

The traditional machine learning approach to wine classification shows interesting but mixed results. Using standard NLP techniques and two classic models, we achieved moderate success: Logistic Regression showed around 79% accuracy, surprisingly outperforming the Random Forest model which achieved 71%. However, these headline figures mask significant variations in performance across different wine origins. Both models showed a strong bias towards US wines (the majority class), achieving their best performance here with an F1-score of 86% for Logistic Regression. The models struggled more with European wines, particularly with the smaller classes like Spanish wines, where we saw high precision but very low recall. French and Italian wines saw moderate performance with F1-scores hovering around 60-70%.

These results highlight several key challenges in traditional ML approaches to this problem. The significant class imbalance in our dataset could be impacting model performance, and while feature engineering helps capture some patterns, the models still struggle with the subtle geographical nuances in wine descriptions. While achieving decent overall accuracy, the inconsistent performance across different countries suggests that more sophisticated approaches might be needed. This limitation of traditional ML/NLP methods in capturing the complex relationships between wine descriptions and their origins provided strong motivation for exploring more advanced approaches like Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

#### Notes on LLM-Based Approaches

The application of Large Language Models (LLMs) to the wine classification task revealed some interesting insights about both the models and the nature of the problem itself. The basic LLM approach, without any Chain-of-Thought (CoT) reasoning or Retrieval-Augmented Generation (RAG), surprisingly achieved the best performance with an accuracy around 85%. This outperformed both the traditional ML approaches and, interestingly, the seemingly more sophisticated LLM variants.

The superior performance of the simple LLM approach suggests several key insights. First, it indicates that the base LLM (GPT-3.5/4) already possesses substantial knowledge about wines and their geographical origins, likely from its training data. The fact that adding RAG or CoT drastically decreases it performance suggests that for this particular task, additional context or structure reasoning might actually be introducing noise rather than helpful information.

This pattern challenges our initial assumption that more context (RAG) or explicit reasoning (CoT) would lead to better results. Instead, it suggests that wine origin classification might be more of a pattern recognition task than a reasoning task, something that the base LLM is already capable enough to handle. The model appears to be picking up on linguistic and descriptive patterns in wine descriptions that directly map to their countries of origin, without needing additional context or explicit reasoning steps.