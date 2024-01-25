from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
load_dotenv() # make it viable in the system


llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectorb_file_path="faiss_index"

# TO store the vectordb into disk
def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt', encoding='latin-1')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectorb_file_path)
# e = embeddings.embed_query("What is your refund policy")
    

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectorb_file_path, instructor_embeddings)
    
    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template="""Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm, 
            chain_type="stuff",
            retriever=retriever,
            input_key="query", 
            return_source_documents=True,
            chain_type_kwargs={"prompt":PROMPT}
           )
    return chain

if __name__ == "__main__":
    # create_vector_db()

    chain = get_qa_chain()
    print(chain("Do you provide internship? Do you have EMI option?"))