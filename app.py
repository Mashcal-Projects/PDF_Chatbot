
import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd
import logging
import re
import matplotlib
import matplotlib.pyplot as plt

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets['OPENAI_API_KEY']

PDF_FILE_PATH = "data/knowledge_center.pdf"

# Example row from your CSV
row = {
    "question": "What are the issues?",
    "diagram": "categories = [מפגע כביש,מפגע מדרכה,מפגע ריהוט,מפגע תברואה,מפגע תמרור]values = [490,467,1,6,1]"
}
# Ensure matplotlib supports RTL languages
matplotlib.rcParams['axes.unicode_minus'] = False  
matplotlib.rcParams['font.family'] = 'Arial' 

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)

# Test if logging works by adding an initial log message
logging.info("App started, logging is set up.")


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

# Function to reverse Hebrew text in each category
def reverse_hebrew_text(categories):
    return [cat[::-1] for cat in categories]
    

def generate_response(prompt, diagram_data=None):
    try:
        with st.spinner("חושב..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "אתה עוזר אדיב, אנא ענה בעברית."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message['content'].strip()
            logging.info(f"answer : {answer}")
            fig = None
            if diagram_data:
                logging.info(f"Diagram data received: {diagram_data}")
                categories, values = parse_diagram_data(diagram_data)

                # Reverse the Hebrew text within each category
                categories = reverse_hebrew_text(categories)
                
                # Log parsed data for further inspection
                if categories and values:
                    try:
                        logging.info(f"Parsed categories: {categories}")
                        fig, ax = plt.subplots(figsize=(3,2))  
                        ax.bar(categories, values)
                      
                        # Rotate the x-axis labels and set the font size smaller
                        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=5)
                        ax.set_yticklabels(categories, rotation=45, ha='right', fontsize=5)
                    except Exception as e:
                        logging.error(f"Error generating graph: {e}")
                else:
                    logging.error("Failed to parse diagram data.")
            
            return answer, fig
            
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error generating response: {e}")
        return None, None
        
# def load_questions(file_path):
#     # Load the questions from a CSV file
#     df = pd.read_csv(file_path)
#     # Assuming questions are in a column named 'Questions'
#     return df['questions'].tolist()
    
def load_questions(file_path):
    # Load the questions and diagrams from a CSV file
    df = pd.read_csv(file_path)
    return df


def user_input(user_question, diagram_data=None):
    # Load the vector store and perform a similarity search
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Use the content of the documents to form a context
    context = " ".join([doc.page_content for doc in docs])

    # Combine the context with the user question and generate a response
    prompt = f"הקשר: {context}\nשאלה: {user_question}\nתשובה:"


    
    # response, diagram = generate_response(prompt, row["diagram"])
    response, diagram = generate_response(prompt, diagram_data)
    st.write(response)
    return  response, diagram


def parse_diagram_data(diagram_str):
    # Extract categories and values using regular expressions
    categories_part = re.search(r'categories = \[(.*?)\]', diagram_str).group(1)
    values_part = re.search(r'values = \[(.*?)\]', diagram_str).group(1)

    # Convert the strings to lists
    categories = categories_part.split(',')
    logging.info(f"categories: {categories}")
    values = list(map(int, values_part.split(',')))
    
    return categories, values
    
def main():
    st.set_page_config("Chat PDF")
    st.markdown(
          """
        <style>
        body {
            direction: rtl;
            text-align: right;
        }
        .st-dr{
            direction: rtl;
            text-align: right;
        }
        .st-e7{
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
    # if 'questions_displayed' not in st.session_state:
    #     st.session_state.questions_displayed = 5 
    # if 'show_more' not in st.session_state:
    #     st.session_state.show_more = False  # Toggle state for showing more questions

    questions_df = load_questions('data/knowledge_center.csv')
    questions = questions_df['questions'].tolist()
     # Input field for custom questions
    user_question = st.text_input("הזינ/י שאלתך (חיפוש חופשי)", key="text_input")

    # Dropdown for predefined questions
    selected_question = st.selectbox("אנא בחר/י מתבנית החיפוש", options=["בחר שאלה..."] + questions)


    
  # Process dropdown selection
    if selected_question != "בחר שאלה...":
        row = questions_df[questions_df['questions'] == selected_question].iloc[0]
        diagram_data = row["diagram"] if pd.notna(row["diagram"]) else None

        if 'last_processed_dropdown' not in st.session_state or st.session_state['last_processed_dropdown'] != selected_question:
            st.session_state['last_processed_dropdown'] = selected_question
            response,diagram = user_input(selected_question,diagram_data)
            logging.info(f"response: {response}, diagram: {diagram}")
            st.session_state.chat_history.append({'question': selected_question, 'answer': response,'diagram':diagram})
            st.rerun()

    # Process custom question input
    if user_question and (user_question != st.session_state.get('last_processed_text', '')):
        row = questions_df[questions_df['questions'] == selected_question].iloc[0]
        diagram_data = row["diagram"] if pd.notna(row["diagram"]) else None
        
        response,diagram = user_input(user_question,diagram_data)
        logging.info(f"response1: {response}, diagram1: {diagram}")
        st.session_state.chat_history.append({'question': user_question, 'answer': response, 'diagram':diagram})
        st.session_state.last_processed_text = user_question
        st.rerun()

    # Display the chat history
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            st.write(f"**שאלה:** {entry['question']}")
            if entry.get('diagram'):  # Safely check for 'diagram' key
                st.pyplot(entry['diagram'])
            st.write(f"**תשובה:** {entry['answer']}")
            st.write("---")  # Separator line

  # Load the vector store (initialization, not directly related to user interaction)
    with st.spinner("מעמיס נתונים..."):
        raw_text = get_pdf_text(PDF_FILE_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
   


if __name__ == "__main__":
    main()
