from dotenv import load_dotenv
from langchain.llms import GooglePalm
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



load_dotenv()




llm = GooglePalm(google_api_key=os.environ["GOOGLE_APIKEY"],temperature=0)



instructor_emb = HuggingFaceInstructEmbeddings()
vector_file_path ="faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="codebasics_faqs.csv",source_column="prompt",encoding='cp1252')
    data=loader.load()
    vector_db = FAISS.from_documents(documents=data,embedding=instructor_emb)
    vector_db.save_local(vector_file_path)



def get_QA_chain():
    vector_db = FAISS.load_local(vector_file_path,instructor_emb)
    retriver = vector_db.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(llm=llm,
    chain_type="stuff",
    retriever=retriver,
    input_key="query",
    return_source_documents=True,chain_type_kwargs=chain_type_kwargs
)
    return chain






if __name__ == "__main__":
    chain = get_QA_chain()
    #print(chain("Do you have javascript cousre?"))