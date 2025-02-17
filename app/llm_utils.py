import os
from dotenv import load_dotenv
# from langchain.llms import OpenAI  # OLD - Remove this line
from langchain_community.chat_models import ChatOpenAI  # NEW - Add this line
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate # Add ChatPromptTemplate
from langchain.chains import LLMChain
from app.utils import chunk_text
from langchain.schema import HumanMessage #For creating message list
import subprocess
import os
import re

def load_llm(model_name, api_key=None, temperature=0):
    """Loads an LLM instance."""
    load_dotenv()
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")  # Or your env variable

    if api_key:
        # Use ChatOpenAI for chat models
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=temperature)
    else:
        raise ValueError("API Key is needed")

def summarize_paper(llm, paper_text, prompt_template=None):
    """Summarizes a research paper using an LLM."""

    docs = [Document(page_content=t) for t in chunk_text(paper_text, 1000, 50)]

    if prompt_template:
        # Use ChatPromptTemplate for chat models
        PROMPT = ChatPromptTemplate.from_template(prompt_template)
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type="map_reduce")

    summary = chain.run(docs)
    return summary

def extract_key_ideas(llm, paper_text, prompt_template=None):
    """Extracts key ideas, contributions, and the central thesis from a paper."""

    if prompt_template is None:
        prompt_template = """
        Identify the key contributions, methodologies, and the central thesis of the following research paper:
        {text}

        Provide a concise summary of each of the following:
        - Key Contributions:
        - Methodologies:
        - Central Thesis:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    ideas = llm_chain.invoke({"text": paper_text}) # Use invoke
    return ideas

def generate_blog_post(llm, summary, key_ideas, prompt_template=None):
    """Generates a blog post based on the paper's summary and key ideas."""

    if prompt_template is None:
        prompt_template = """
        Write a blog post explaining the following research paper in an accessible way for a general audience with some technical background.
        ... (rest of your prompt template) ...
        Blog Post:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    blog_post = llm_chain.invoke({"summary": summary, "key_ideas": key_ideas}) #Pass directly
    return blog_post

def generate_video_script(llm, summary, key_ideas, prompt_template=None):
    """Generates a video script based on the paper's summary and key ideas."""

    if prompt_template is None:
        prompt_template = """
        You are a scriptwriter for educational videos.  Your task is to create a script for a 3-5 minute explainer video about a research paper.  The video is aimed at a general audience with some technical background.

        Here is a summary of the research paper:

        {summary}

        Here are the key ideas and methodology from the paper:

        {key_ideas}

        Your script MUST follow this structure EXACTLY:

        1. **INTRODUCTION (0:00-0:30):**
           - Start with a hook to grab the viewer's attention (e.g., a surprising fact, a relevant question, a real-world problem).
           - Briefly introduce the topic of the research paper.
           - State the main finding of the paper in a single sentence.

        2. **CORE CONCEPTS (0:30-2:00):**
           - Explain the core concepts and methods used in the research.
           - Use clear, simple language. Avoid jargon as much as possible.
           - Break down complex ideas into smaller, digestible steps.
           - Use analogies and real-world examples.
           - Include specific cues for visuals.  Use the following format for visual cues: `[VISUAL: description of visual]`. For instance:
             - `[VISUAL: Diagram of the model architecture]`
             - `[VISUAL: Animation showing the algorithm in action]`
             - `[VISUAL: Graph showing the results]`
             - `[VISUAL: Equation for X]`

        3. **RESULTS AND IMPLICATIONS (2:00-3:30):**
           - Describe the main results of the research.
           - Explain the implications of these results.  Why are they important?
           - Discuss any limitations of the research.
           - Include visual cues.

        4. **CONCLUSION (3:30-4:00):**
           - Briefly summarize the key takeaways.
           - Suggest potential future research directions.
           - End with a call to action (e.g., "Read the full paper here: [link]", "Subscribe for more science videos!").

        5. **END SCREEN (4:00-4:30):**
           - [VISUAL: End screen with title, author, and call to subscribe/follow]

        Write the script, including the narration text and the visual cues.  Be concise, engaging, and informative.  Maintain an objective tone. Do NOT include timings in your output; ONLY include the narration text and visual cues.

        Video Script:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    video_script = llm_chain.invoke({"summary": summary, "key_ideas": key_ideas})
    return video_script


def generate_manim_code(llm, concept_description, prompt_template=None):
    """Generates Manim code to visualize a given concept."""
    if prompt_template is None:
        prompt_template = """
        You are a Manim code generator. Your task is to generate Manim code to create a short animation based on the following description:

        Visual Cue: {{concept_description}}

        Here are some examples of Manim code:

        # Example 1: Display centered text with specific font and size
        ```python
        from manim import *

        class TempScene(Scene):
            def construct(self):
                text = Text("Hello, Manim!", color=BLUE, font_size=48, font="Arial")
                text.move_to(ORIGIN)
                self.add(text)
        self.wait(1)
        # Example 2: Draw a circle and square
        
        from manim import *

        class TempScene(Scene):
            def construct(self):
                text = Text("Title", color=WHITE, font_size=60)
                text.to_edge(UP) # Move to the top edge
                self.play(FadeIn(text, run_time=2)) #Fade in
                self.wait(1)
        

        # Example 3: Show a simple equation
        
        from manim import *

        class TempScene(Scene):
            def construct(self):
                title = Text("My Diagram", font_size=48, color=YELLOW)
                title.to_edge(UP)
                self.play(Write(title))

                circle = Circle(color=RED)
                square = Square(color=GREEN)
                square.next_to(circle, RIGHT)

                self.play(Create(circle))
                self.play(Create(square))
                self.wait(1)
        
        Generate Manim code that:
        - Creates a class called `TempScene` that inherits from `Scene`.
        - Includes all necessary imports (e.g., `from manim import *`).
        - Implements the `construct` method.
        - Creates the visualization described by the "Visual Cue" above.
        - Uses appropriate Manim objects and animations.
        - Is well-commented and easy to understand.
        ABSOLUTELY NO EXTRA TEXT OR EXPLANATIONS!
        OUTPUT ONLY VALID PYTHON CODE.
        DO NOT INCLUDE ANY MARKDOWN CODE BLOCK DELIMITERS (```).
        DO NOT INCLUDE ANY INTRODUCTORY OR CONCLUDING PHRASES.
        ONLY PYTHON CODE

        Manim Code:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    manim_code = llm_chain.invoke({"concept_description": concept_description})
    # --- Add post-processing to remove extraneous text ---
    manim_code_str = manim_code['text']
    manim_code_str = manim_code_str.replace("`python", "").replace("`", "").strip()

    # Additional cleanup: Find the first 'from manim' and keep only that part
    start_index = manim_code_str.find("from manim")
    if start_index != -1:
        manim_code_str = manim_code_str[start_index:]

    return {"text": manim_code_str} # Return the cleaned code




import subprocess
import os
import re
import glob

def run_manim_code(manim_code, scene_name="TempScene"):
    """
    Runs the given Manim code and returns the path to the output video file.
    Handles potential errors during Manim execution.
    """

    temp_file_path = f"temp_manim_scene_{scene_name}.py"
    with open(temp_file_path, "w") as f:
        f.write(manim_code)

    # --- Extract Scene Name Using Regex ---
    match = re.search(r"class\s+([a-zA-Z0-9_]+)\s*\(.*Scene.*\):", manim_code)
    if match:
        extracted_scene_name = match.group(1)
    else:
        raise ValueError("Could not find Scene class name in Manim code.")
    # --- End Regex Extraction ---

    command = [
        "manim",
        temp_file_path,
        extracted_scene_name,  # Use extracted_scene_name
        "-ql",
        "--media_dir", "./media"
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print(process.stdout)
        video_file_name = f"{extracted_scene_name}.mp4"

        # Find video file using glob, much safer.
        video_file_path = os.path.join("media", "videos", temp_file_path.replace(".py", ""), "480p15", "**", video_file_name)
        video_files = glob.glob(video_file_path, recursive=True)

        if not video_files: #if list is empty
            raise FileNotFoundError(f"Manim output video not found for scene {scene_name}")
        video_file_path = video_files[0] #Get the first match
        #Make it relative
        video_file_path = os.path.relpath(video_file_path, start=os.path.dirname(__file__))
        video_file_path = video_file_path.replace("\\","/")
        return video_file_path

    except subprocess.CalledProcessError as e:
        print(f"Manim command: {e.cmd}")
        print(f"Manim stdout: {e.stdout}")
        print(f"Manim stderr: {e.stderr}")
        raise e
    except FileNotFoundError as e:
        raise e
    finally:
        try:
            os.remove(temp_file_path)
        except OSError:
            pass










