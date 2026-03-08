from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DB_PATH = "vectorstore"


def ask_question(question):

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt_template = """
You are an AI assistant answering questions about uploaded documents.

Use ONLY the document context to answer.

If the question asks for:
- summary → summarize
- explanation → explain clearly
- list → extract key points

If the answer is not in the documents say:
"I could not find this in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})

    answer = result["result"]

    sources = []

    for doc in result["source_documents"]:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "document")

        sources.append(f"{source} (page {page})")

    return answer, sources
