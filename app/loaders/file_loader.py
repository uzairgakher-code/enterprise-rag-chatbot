from langchain.document_loaders import DirectoryLoader, TextLoader

def load_files(path):
    loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    return loader.load()
