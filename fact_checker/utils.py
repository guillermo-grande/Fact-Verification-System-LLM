import os
def load_prompt(type: str, version: int) -> str:
    """Given the type of a prompt and its version, this function returns the prompt. 

    Args:
        type (str): prompt type. ejem. decompose, generate, claim
        version (int): prompt version

    Returns:
        str: text of the prompt in the file `{type}/{type}-v{version}.txt`
    """
    file_name = f"./prompts/{type}/{type}-v{version}.txt"
    if not os.path.exists(file_name):
        raise FileNotFoundError
    
    with open(file_name, 'r') as prompt_file:
        content = prompt_file.read()
    return content