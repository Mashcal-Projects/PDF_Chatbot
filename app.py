import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd

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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "אתה עוזר אדיב, אנא ענה בעברית."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def load_questions(file_path):
    # Load the questions from a CSV file
    df = pd.read_csv(file_path)
    # Assuming questions are in a column named 'Questions'
    return df['questions'].tolist()

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

    # st.write(response)
    return response
    
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
     # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    questions = load_questions('data/knowledge_center.csv')
    
      # Text input for user's question
     # Manage input field with state control
    user_question = st.text_input("שאל אותי הכל!", value=st.session_state.get('user_input', ''))

    # Display buttons for predefined questions
    cols = st.columns(5)
    for i, question in enumerate(questions[:5]):  # Limiting to first 5 questions for simplicity
        if cols[i % 5].button(question):
            st.session_state['user_input'] = question  # Update session state with the selected question - Added 
            with st.spinner("חושב..."):  # Add spinner here
                response = user_input(question)  # Generate the response
            st.session_state.chat_history.append({'question': question, 'answer': response})
            st.session_state['last_processed'] = question  # Track last processed question
            st.experimental_rerun() 
   
        # Process input (either from text input or button selection)
    if user_question and (user_question != st.session_state.get('last_processed', '')):
        response = user_input(user_question)  # Generate the response
        st.session_state.chat_history.append({'question': user_question, 'answer': response})
        st.session_state['last_processed'] = user_question  # Track last processed question
        st.experimental_rerun()  # Rerun to display the updated chat history
        
        
        # Display the chat history
    if st.session_state.chat_history:
        # st.write("## היסטוריית צ'אט")
        for entry in st.session_state.chat_history:
            st.write(f"**שאלה:** {entry['question']}")
            st.write(f"**תשובה:** {entry['answer']}")
            st.write("---")  # Separator line

    with st.spinner("חושב..."):
        raw_text = get_pdf_text(PDF_FILE_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
    
    
    # categories, values = process_pdf(PDF_FILE_PATH)

    # if categories and values:
    #     fig, ax = plt.subplots()
    #     ax.bar(categories, values)
    #     # ax.set_xlabel('Categories')
    #     # ax.set_ylabel('Values')
    #     # ax.set_title('Bar Chart from PDF Data')

    #     st.pyplot(fig)
    # else:
    #     st.write("Categories or values not found in the PDF.")


if __name__ == "__main__":
    main()
