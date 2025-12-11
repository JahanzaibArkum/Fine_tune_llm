from llama_cpp import Llama
import logging

# Completely silence llama logs
logging.getLogger("llama_cpp").setLevel(logging.CRITICAL)

# Load model with safe settings
llm = Llama(
    model_path="models/open-llama-3b/open-llama-3b-v2-wizard-evol-instuct-v2-196k.Q3_K_M.gguf",
    n_ctx=2048,
    temperature=0.0,     # 🔒 disables creative hallucination
    top_p=0.1,
    verbose=False
)

# ✅ STRONG FEW-SHOT + STRICT FORMAT PROMPT
prompt = """
You are a grammar correction machine.
You ONLY return the corrected sentence.
Never explain anything.

Example:
Input: She go to school.
Output: She goes to school.

Input: I has a apple.
Output:
"""

# ✅ STOP generation after first line
output = llm(
    prompt,
    max_tokens=200
    
)

print(output["choices"][0]["text"].strip())
