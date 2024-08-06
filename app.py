import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

GOOGLE_API_KEY= st.secrets['GOOGLE_API_KEY']

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

PDF_FILE_PATH = "data/knowledge_center.pdf" 




def get_pdf_text(pdf_file_path):
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")


# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.2)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain



# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization= True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

    
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write(response["output_text"])






def get_vector_store(text_chunks):
    from langchain.embeddings import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def generate_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "אתה עוזר אדיב, אנא ענה בעברית."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def user_input(user_question):
    # Load the vector store and perform a similarity search
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Use the content of the documents to form a context
    context = " ".join([doc.page_content for doc in docs])

    # Combine the context with the user question and generate a response
    prompt = f"הקשר: {context}\nשאלה: {user_question}\nתשובה:"
    response = generate_response(prompt)

    st.write(response)



def main():
    st.set_page_config("Chat PDF")
    st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.header("מודל שפה משכ״ל🤖🗨️")

    user_question = st.text_input("שאל אותי הכל!")

    if user_question:
        user_input(user_question)

    with st.spinner("חושב..."):
        raw_text = get_pdf_text(PDF_FILE_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Done")

if __name__ == "__main__":
    main()
