SYSTEM_PROMPT = """

A conversation between User and Assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer (as A, B, C, D). The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here as a letter </answer>.



"""

def get_system_prompt():
    return SYSTEM_PROMPT
