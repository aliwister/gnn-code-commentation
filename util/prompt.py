import re
import pdb

def extract_first_user_comment(text):
    # Define the regular expression pattern
    pattern = r'User\'s Comment:\s*"(.*?)"'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the first captured group (the text inside the double quotes)
    if match:
        return match.group(1)
    else:
        # If no match is found, return None or a custom message
        return None
    
def find_quoted_text_after_3rd_output(text):
    # Define a regular expression pattern to match "Output:" followed by quoted text
    pattern = r'Output:\s+"(.*?)"'
    
    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)
    
    # Check if there are at least 3 matches
    if len(matches) >= 3:
        # Return the quoted text after the 3rd "Output:"
        return matches[2]
    
    # Return None if there are fewer than 3 matches
    return None


def get_answer(prompt, lang_model, tokenizer, device):
    input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    outputs = lang_model.generate(input_ids, attention_mask=attention_mask, max_length=2048,  A do_sample=True,)
    
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print(answer)
    #pdb.set_trace()
    #print(answer)
    #parsed = [re.findall(r'Output:[\s\S]+?"(.*?)"', a)[2] for a in answer]
    parsed =  [re.search(r'###.*?Output:(.*?)(?:$|\n)', a, re.DOTALL).group(1) for a in answer]
    print(parsed)
    return parsed

def create_zeroshot_prompt(arg):
    return """Instructions:
    You are an assistant that comments on text-based queries. For each query provided, return a 1-line comment to explain what the code does.


    User's Code: "{0}"
    Comment: 
    """.format(arg)

def create_incontext_prompt(*args):
    return """
    Task: You are an assistant that comments on text-based queries. For each query provided, return a 1-line comment to explain what the code does. The comments should provide the question the query is solving.
    Examples:
    Input: "{0}"
    Output: "{1}"
    Input: "{2}"
    Output: "{3}"
    ###
    Input: "{4}"
    Output: """.format(*args)