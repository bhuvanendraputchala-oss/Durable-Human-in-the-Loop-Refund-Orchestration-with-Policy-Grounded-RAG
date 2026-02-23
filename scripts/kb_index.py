import os
from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

POLICY_DIR = "mock_data/policies"
PERSIST_DIR = "vector_db"


def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )

    texts = []
    metadatas = []

    for file in sorted(os.listdir(POLICY_DIR)):
        if not file.endswith(".md"):
            continue

        with open(f"{POLICY_DIR}/{file}", encoding="utf-8") as f:
            content = f.read()

        for idx, doc in enumerate(splitter.create_documents([content])):
            texts.append(doc.page_content)
            metadatas.append({
                "source": file,
                "chunk_id": f"{file}:chunk_{idx}",
                "start_char": doc.metadata.get("start_index", 0),
            })

    Chroma.from_texts(
        texts,
        OpenAIEmbeddings(),
        metadatas=metadatas,
        collection_name="policy_kb",
        persist_directory=PERSIST_DIR,
    )
    print(f"Indexed {len(texts)} chunks from {POLICY_DIR} into {PERSIST_DIR}.")


if __name__ == "__main__":
    main()
