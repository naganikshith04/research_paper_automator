# test_manim_prompt.py
from app.llm_utils import load_llm, generate_manim_code

llm = load_llm("gpt-4o")  # Use the same model as your main script

# A simplified version of your main loop, just for testing Manim
test_concept = "Display the text 'AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents' in blue, centered on the screen." # Or put the summary here
manim_code = generate_manim_code(llm, test_concept)
print(manim_code['text'])