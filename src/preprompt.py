from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import json


def parse_json_from_text(response_text):
    """
    Extracts and parses JSON content from a formatted string with markdown-like backticks.
    """
    # Extract the JSON part from the response using regex
    match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
    if match:
        json_text = match.group(1).strip()  # Extract the JSON text within the backticks
        try:
            # Parse the JSON content
            data = json.loads(json_text)
            return data
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            return None
    else:
        print("No JSON found in the text.")
        return None

def extract_components_and_assembly(prompt):
    """
    Uses OpenAI GPT model to extract components, dimensions, and assembly instructions from a design prompt.
    Returns a dictionary with structured information.
    """
    # Load API key from .env file
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="https://api.perplexity.ai",  # This is the default and can be omitted
    )

    # Define the prompt for the GPT model
    gpt_prompt = (
        f"""
        Analyze the following design prompt and extract:
        1. A structured map of components and their dimensions.
        2. A separate set of instructions for assembly.
        
        Input Prompt:
        {prompt}

        Output Format:
        {{
            "components": {{
                "component_name": "dimensions",
                ...
            }},
            "assemble": "assembly instructions"
        }}

        Only give me the output json without appending anything to it.
        """
    )

    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": gpt_prompt},
                ],
            }
        ],
    )

    # Parse the response
    return response.choices[0].message.content

# Example usage
prompt = (
    "Design a ceiling fan with a 200 mm diameter, 100 mm tall motor housing, and four 500 mm long aluminum blades. "
    "Attach a 300 mm steel suspension rod to the top of the housing and a 100 mm ceiling mount at the other end. "
    "Assemble by securing the blades to the housing and the rod to the mount. Ensure balance and export as STEP files."
)

result = extract_components_and_assembly(prompt)
print(result)
