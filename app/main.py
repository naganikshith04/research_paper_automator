from app.utils import fetch_paper, extract_text_from_pdf, extract_text_from_html
from app.llm_utils import load_llm, summarize_paper, extract_key_ideas, generate_blog_post, generate_video_script, generate_manim_code, run_manim_code
import subprocess
import re
from moviepy.editor import concatenate_videoclips, VideoFileClip, AudioFileClip #add AudioFileClip
from gtts import gTTS #Import gTTS

def main():
    # ... (rest of your main.py code up to video script generation) ...
    # 2. Fetch and preprocess the paper
    paper_input = input("Enter the URL or title of the research paper: ")
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
    print("\nSummary:\n", summary)

    # 5. Generate content
    blog_post = generate_blog_post(llm, summary, key_ideas['text'])
    video_script = generate_video_script(llm, summary, key_ideas['text'])

    # --- Parse Video Script for Visual Cues ---
    visual_cues = []
    for line in video_script['text'].splitlines():
        match = re.search(r"\[VISUAL:\s*(.*?)\s*\]", line)
        if match:
            visual_cues.append(match.group(1))

    print("\nExtracted Visual Cues:\n", visual_cues)

    # --- Generate and Run Manim Code for EACH Cue ---
    video_file_paths = []
    for i, cue in enumerate(visual_cues):
        manim_code = generate_manim_code(llm, cue)
        print(f"\nManim Code for Cue '{cue}':\n", manim_code['text'])

        try:
            video_file_path = run_manim_code(manim_code['text'], scene_name=f"TempScene_{i}")
            print(f"Manim video generated: {video_file_path}")
            video_file_paths.append(video_file_path)
        except subprocess.CalledProcessError as e:
            print(f"Error running Manim for cue '{cue}':")
            print(f"  Manim command: {e.cmd}")
            print(f"  Manim stdout: {e.stdout}")
            print(f"  Manim stderr: {e.stderr}")
        except FileNotFoundError as e:
            print(f"Error running Manim for cue '{cue}': {e}")
        except Exception as e:
            print(f"Error on Manim execution: {e}")
    # --- Concatenate Videos and Add Audio ---
    if video_file_paths:
        clips = [VideoFileClip(path) for path in video_file_paths]
        final_clip = concatenate_videoclips(clips)

        # Generate audio from the video script
        tts = gTTS(text=video_script['text'], lang='en')  # Adjust language if needed
        audio_file = "narration.mp3"
        tts.save(audio_file)

        # Add audio to the video
        audio_clip = AudioFileClip(audio_file)
        final_clip = final_clip.set_audio(audio_clip)


        final_video_path = "final_video.mp4"
        final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
        print(f"Final video generated: {final_video_path}")

        # Clean up the temporary audio file (optional)
        try:
            os.remove(audio_file)
        except OSError:
            pass
    # ... (rest of your main.py code) ...
    # 6. Present content to the user
    print("\nBlog Post:\n", blog_post['text'])
    print("\nVideo Script:\n", video_script['text'])

    # 7. Get user approval
    approval = input("Approve content? (y/n): ")

    if approval.lower() == 'y':
        # 8. Publish content
        print("Content published!")
    else:
        print("Content not published.")

if __name__ == "__main__":
    main()