import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from io import BytesIO
import arxiv
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PaperNotFoundError(Exception):
    pass


def fetch_paper(url_or_title):
    """Fetches the content of a research paper given its URL or title."""
    try:
        if url_or_title.startswith("http"):  # It's a URL
            response = requests.get(url_or_title)
            response.raise_for_status()  # Raise an exception for bad status codes

            if "application/pdf" in response.headers.get("Content-Type", "").lower():
                return response.content  # Return raw bytes for PDF
            elif "text/html" in response.headers.get("Content-Type", "").lower():
                return response.text  # Return text for HTML
            else:
                raise ValueError("Unsupported content type: {}".format(response.headers.get("Content-Type")))

        else:  # It's a title, try searching on arXiv
            search = arxiv.Search(query=url_or_title, max_results=1)
            results = list(search.results())  # Force evaluation of the iterator
            if results:
                paper = results[0]
                response = requests.get(paper.pdf_url)
                response.raise_for_status()
                return response.content  # Return raw bytes (it's a PDF)
            else:
                raise PaperNotFoundError(f"No paper found for title: '{url_or_title}'")

    except requests.exceptions.RequestException as e:
        raise PaperNotFoundError(f"Error fetching paper: {e}") from e


def extract_text_from_pdf(pdf_content):
    """Extracts text from PDF content (bytes)."""
    with BytesIO(pdf_content) as f:
        text = extract_text(f)
    return text


def extract_text_from_html(html_content):
    """Extracts text from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    # This is a basic example; you might need to refine the selection
    # to target specific elements containing the main paper text.
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

def chunk_text(text, chunk_size, overlap):
    """Splits the text into chunks with a specified size and overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks