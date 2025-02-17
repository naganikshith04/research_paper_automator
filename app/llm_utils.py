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
        You are a Manim code generator. Your ONLY task is to generate Manim code to display the following text on the screen:

        Text: {concept_description}

        Generate Manim code that:

        - Creates a class called `TempScene` that inherits from `Scene`.
        - Includes all necessary imports (e.g., `from manim import *`).
        - Implements the `construct` method.
        - Uses the `Text` class to display the text.
        - Centers the text on the screen. Use `text.move_to(ORIGIN)`.
        - Sets the text color to blue. Use `color=BLUE`.
        - Includes a `self.add(text)` line to add the text to the scene.
        - Includes a `self.wait(1)` line to pause the scene for 1 second.

        Here's an example of Manim code that displays text:
        ```python
        from manim import *

        class TempScene(Scene):
            def construct(self):
                text = Text("Hello, Manim!", color=BLUE)
                text.move_to(ORIGIN)
                self.add(text)
                self.wait(1)

        # Example 2: Draw a circle and square
        ```python
        from manim import *

        class TempScene(Scene):
            def construct(self):
                circle = Circle()
                square = Square()
                self.play(Create(circle))
                self.play(Transform(circle, square))
                self.wait(1)
        ```

        # Example 3: Show a simple equation
        ```python
        from manim import *

        class TempScene(Scene):
            def construct(self):
                equation = MathTex(r"E=mc^2")
                self.play(Write(equation))
                self.wait(1)
        ```
        Generate Manim code that:
        - Creates a class called `TempScene` that inherits from `Scene`.
        - Includes all necessary imports (e.g., `from manim import *`).
        - Implements the `construct` method.
        - Creates the visualization described by the "Visual Cue" above.
        - Uses appropriate Manim objects and animations.
        - Is well-commented and easy to understand.

        Manim Code:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    manim_code = llm_chain.invoke({"concept_description": concept_description})
    return manim_code


def run_manim_code(manim_code):
    """
    Runs the given Manim code and returns the path to the output video file.
    Handles potential errors during Manim execution.
    """

    # 1. Create a temporary Python file
    temp_file_path = "temp_manim_scene.py"
    with open(temp_file_path, "w") as f:
        f.write(manim_code)

    # 2. Construct the Manim command (extract scene name and use it)
    match = re.search(r"class\s+([a-zA-Z0-9_]+)\s*\(.*Scene.*\):", manim_code)
    if match:
        scene_name = match.group(1)
    else:
        raise ValueError("Could not find Scene class name in Manim code.")

    command = [
        "manim",
        temp_file_path,
        scene_name,  # Use the extracted scene name
        "-ql",         # Low quality for faster rendering during development
        "--media_dir", "./media" #specify media directory
    ]

    # 3. Run Manim using subprocess
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print(process.stdout)

        # 4. Determine the output video file path
        video_file_name = f"{scene_name}.mp4" # use scene name
        video_file_path = os.path.join("media", "videos", temp_file_path.replace(".py",""), "480p15", video_file_name)


        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Manim output video not found at {video_file_path}")
        return video_file_path

    except subprocess.CalledProcessError as e:
        error_message = f"Manim execution failed:\n{e.stderr}"
        raise e # Raise the subprocess error
    except FileNotFoundError as e:
        raise e
    finally:
        # 5. Clean up the temporary file
        try:
            os.remove(temp_file_path)
        except OSError:
            pass

