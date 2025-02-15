from app.utils import fetch_paper, extract_text_from_pdf, extract_text_from_html
from app.llm_utils import load_llm, summarize_paper, extract_key_ideas, generate_blog_post, generate_video_script, generate_manim_code, run_manim_code 
import subprocess

def main():
    # 1. Get user input
    paper_input = input("Enter the URL or title of the research paper: ")

    # 2. Fetch and preprocess the paper
    try:
        paper_content = fetch_paper(paper_input)
        if paper_content.startswith(b"%PDF"):  # Check if it's a PDF
            paper_text = extract_text_from_pdf(paper_content)
        else:
            paper_text = extract_text_from_html(paper_content.decode('utf-8'))
    except Exception as e:
        print(f"Error fetching/processing paper: {e}")
        return

    # 3. Load the LLM
    llm = load_llm("gpt-4o")

    # 4. Summarize and extract key ideas
    summary = summarize_paper(llm, paper_text)
    key_ideas = extract_key_ideas(llm, paper_text)
    print("\nSummary:\n", summary)  # Access summary directly

   # 5. Generate content
    blog_post = generate_blog_post(llm, summary, key_ideas['text'])  # Access the 'text' key
    video_script = generate_video_script(llm, summary, key_ideas['text'])  # Access the 'text' key

    # --- Generate Manim code ---
    manim_code = generate_manim_code(llm, summary)  # Pass the summary
    print("\nManim Code:\n", manim_code['text']) # manim_code has content


    try:
        video_file_path = run_manim_code(manim_code['text']) #now it has content attribute
        print(f"Manim video generated: {video_file_path}")
    except Exception as e:
        print(f"Error running Manim: {e}")


    # 6. Present content to the user (you'll need to build a UI for this)
    print("\nBlog Post:\n", blog_post['text'])  # Access blog_post directly
    print("\nVideo Script:\n", video_script['text'])  # Access video_script directly

    # 7. Get user approval
    approval = input("Approve content? (y/n): ")

    if approval.lower() == 'y':
        # 8. Publish content (implement in app/publication.py)
        # ... (call functions to post the blog and upload the video) ...
        print("Content published!")
    else:
        print("Content not published.")

if __name__ == "__main__":
    main()