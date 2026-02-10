from langchain.document_loaders import WebBaseLoader

def load_urls(url_file):
    with open(url_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    loader = WebBaseLoader(urls)
    return loader.load()
