GUESS_WINE = """
Based on the following wine information, guess the country of origin from these options: 
US, France, Italy, Spain.

Wine information:
{formatted_text}

You must respond like so:
Answer: country
"""



GUESS_WINE_COT = """
Based on the following wine information, guess the country of origin from these options: 
US, France, Italy, Spain.

Wine information:
{formatted_text}

Think about the wine and respond with some of your thoughts before giving answer:
Ansewr should be given in this form:
Answer: country
"""

RAG_WINE = """
Based on the following wine information, guess the country of origin from these options: 
US, France, Italy, Spain.

Here are some similar wine examples from our database:

SIMILAR WINES:
{similar_wines}

WINE TO CLASSIFY:
{formatted_text}

You must respond like so:
Answer: country
"""

RAG_WINE_COT = """
Based on the following wine information, guess the country of origin from these options: 
US, France, Italy, Spain.

Here are some similar wine examples from our database:

SIMILAR WINES:
{similar_wines}

WINE TO CLASSIFY:
{formatted_text}

Think about how this wine compares to the similar examples and respond with your thoughts before giving answer.
Answer should be given in this form:
Answer: country
"""