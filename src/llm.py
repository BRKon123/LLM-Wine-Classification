from utils.text import format_dict_to_string
from ai_clients.openai_helper import OpenAIHelper
from prompts.prompts import GUESS_WINE, GUESS_WINE_COT
import pandas as pd
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

def guess_wine_country(api_key: str, wine_data_path: str, use_cot: bool = False) -> List[str]:
    """
    Guess the country of origin for each wine entry and track accuracy.
    """
    # Load the wine quality data
    wine_data = pd.read_csv(wine_data_path, index_col = 0)
    test_data = wine_data.iloc[-200:]
    
    # Initialize OpenAIHelper and tracking variables
    openai_helper = OpenAIHelper(api_key=api_key)
    guessed_countries = []
    correct_predictions = 0
    total_predictions = 0
    
    for _, row in test_data.iterrows():
        try:
            total_predictions += 1
            true_country = row['country']
            
            # Remove the 'country' column
            wine_info = row.drop('country').to_dict()
            
            # Format the remaining data
            formatted_text = format_dict_to_string(wine_info)
            prompt = GUESS_WINE_COT if use_cot else GUESS_WINE
            formatted_prompt = prompt.format(formatted_text=formatted_text)
            
            # Prepare messages for OpenAI API
            messages = [
                {"role": "system", "content": "You are a wine expert."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            # Get the guessed country and extract answer
            response = openai_helper.create_chat_completion(messages)
            guessed_country = response.split("Answer:")[-1].strip()
            
            # Track accuracy
            is_correct = guessed_country.lower() == true_country.lower()
            if is_correct:
                correct_predictions += 1
                
            # Calculate current accuracy
            current_accuracy = (correct_predictions / total_predictions) * 100
            
            print(f"Prediction {total_predictions}:")
            print(f"True country: {true_country}")
            print(f"Guessed country: {guessed_country}")
            print(f"Correct: {is_correct}")
            print(f"Current accuracy: {current_accuracy:.2f}%\n")
            
            guessed_countries.append(guessed_country)
            
        except Exception as e:
            print(f"Error occurred in iteration {total_predictions}: {str(e)}")
            guessed_countries.append("ERROR")
            continue
    
    # Print final accuracy
    final_accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nFinal Results:")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Final accuracy: {final_accuracy:.2f}%")
    
    return guessed_countries


if __name__ == "__main__":
    guess_wine_country(
        api_key=os.getenv("ARTIFICIAL_API_KEY"),
        wine_data_path="data/wine_quality_1000.csv",
        use_cot=True
    )