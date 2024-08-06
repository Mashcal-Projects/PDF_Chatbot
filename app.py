import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets['OPENAI_API_KEY']

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

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "אתה עוזר אדיב, אנא ענה בעברית."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def user_input(user_question):
    # Load the vector store and perform a similarity search
    embeddings = OpenAIEmbeddings()
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
