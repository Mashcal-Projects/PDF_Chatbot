
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
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.patches as mpatches
import matplotlib.axes as maxes
from matplotlib.projections import register_projection
from gradio_client import Client, handle_file
from difflib import get_close_matches
import sys

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets['OPENAI_API_KEY']

# PDF_FILE_PATH = "data/sitemap_data.pdf"
PDF_FILE_PATH = "data/knowledge_center.pdf"

# Example row from your CSV
row = {
    "question": "What are the issues?",
    "diagram": "categories = [מפגע כביש,מפגע מדרכה,מפגע ריהוט,מפגע תברואה,מפגע תמרור]values = [490,467,1,6,1]"
}
# Ensure matplotlib supports RTL languages
matplotlib.rcParams['axes.unicode_minus'] = False  

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),]
)

# Test if logging works by adding an initial log message
logging.info("App started, logging is set up.")

def load_embeddings():
    """Loads OpenAI embeddings and stores them in session_state for reuse."""
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        logging.info("OpenAI embeddings initialized and stored in session_state.")


# Load the vector store once and store it in session_state if not already loaded
def load_vector_store():
    if 'vector_store' not in st.session_state:
        with st.spinner("טוען נתונים..."):
            raw_text = get_pdf_text(PDF_FILE_PATH)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            logging.info("Vector store loaded and stored in session_state.")


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
     # Ensure that embeddings are loaded only once
    if 'embeddings' not in st.session_state:
        load_embeddings()
    
    embeddings = st.session_state.embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to reverse Hebrew text in each category
def reverse_hebrew_text(categories):
    return [cat[::-1] for cat in categories]
    
def find_closest_question(user_question, questions_df):
    # Use difflib's get_close_matches to find the closest matching question
    questions = questions_df['questions'].tolist()
    closest_matches = get_close_matches(user_question, questions, n=1, cutoff=0.5)  # Adjust cutoff as needed
    if closest_matches:
        return closest_matches[0]  # Return the closest question
    return None
    

def generate_response(prompt, diagram_data=None):
    try:
        with st.spinner("חושב..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
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
                        # Replace the original part with this
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        ax.spines[["right", "top"]].set_visible(False)
                        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
                        bars = ax.bar(categories, values, label=categories, color=bar_colors)
                        ax.set_ylim(0, max(values) * 1.2)
                        plt.xticks(rotation=45)

                        # Add value labels on top of the bars with a small font size
                        if len(values) > 1:
                            for bar in bars:
                                yval = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval}', ha='center', va='bottom', fontsize=8)
                        ax.legend(loc='upper right',alignment='center')
                        
                        # Use the custom handler to reverse icon and text order
                        # ax.legend(bars, categories, loc='upper right', handler_map={bars[0]: ReversedLegendHandler()})
                    except Exception as e:
                        logging.error(f"Error generating graph: {e}")
                else:
                    logging.error("Failed to parse diagram data.")
            return answer, fig
            
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error generating response: {e}")
        return None, None
        
def load_questions(file_path):
    # Load the questions and diagrams from a CSV file
    df = pd.read_csv(file_path)
    return df

def user_input(user_question, diagram_data=None):
    logging.info(f"user_question: {user_question}")
    # Initialize embeddings only once
    if 'embeddings' not in st.session_state:
        load_embeddings()
    
    embeddings = st.session_state.embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Use the content of the documents to form a context
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"הקשר: {context}\nשאלה: {user_question}\nתשובה קצרה:"
    logging.info(f"prompt: {prompt}")
      # Initialize response and diagram to avoid UnboundLocalError
    response = ""
    response, diagram = generate_response(prompt, diagram_data)
    return response, diagram


def parse_diagram_data(diagram_str):
    logging.info("parse_diagram_data")
    # Extract categories and values using regular expressions
    categories_part = re.search(r'categories = \[(.*?)\]', diagram_str).group(1)
    values_part = re.search(r'values = \[(.*?)\]', diagram_str).group(1)

    # Convert the strings to lists
    categories = categories_part.split(',')
    # logging.info(f"categories: {categories}")
    values = list(map(int, values_part.split(',')))
    return categories, values

def reset_conversation():
    st.session_state.chat_history = []
    
def main():
    st.set_page_config("Chat PDF")
    st.markdown(
        """
        <style>
        body {
            direction: rtl;
            text-align: right;
            # background-color: white;
            # color:black;
        }
        # .st-bb ,h2, p{
        #     background-color: white;
        #     color:black;
        
        # }
        # .stApp {
        #     background-color: white;
        # }
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

    
    st.header("שאל את מומחה התשתיות 🤖🗨️")
     # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load the vector store once
    load_vector_store()
    questions_df = load_questions('data/knowledge_center.csv')
    # questions_df = load_questions('data/sitemap_data.csv')
    questions = questions_df['questions'].tolist()

     # Input field for custom questions
    user_question = st.text_input("הזינ/י שאלתך (חיפוש חופשי)",key="user_question")
    # Dropdown for predefined questions
    selected_question = st.selectbox("אנא בחר/י מתבנית החיפוש", options=["בחר שאלה..."] + questions,key="selected_question")
    
    # Add Reset Button for Conversation
    if st.button("אפס שיחה"):
        reset_conversation()
        
    # Process dropdown selection  
    if selected_question != "בחר שאלה...":
            row = questions_df[questions_df['questions'] == selected_question].iloc[0]
            diagram_data = row["diagram"] if pd.notna(row["diagram"]) else None

            if 'last_processed_dropdown' not in st.session_state or st.session_state['last_processed_dropdown'] != selected_question:
                st.session_state['last_processed_dropdown'] = selected_question
                response,diagram = user_input(selected_question,diagram_data)
                logging.info(f"response: {response}, diagram: {diagram}")
                st.session_state.chat_history.append({'question': selected_question, 'answer': response,'diagram':diagram})
            
    # Process input text
    if user_question and (user_question != st.session_state.get('last_processed', '')):
        st.session_state['last_processed'] = user_question  # Track last processed question
        # closest_question = find_closest_question(user_question, questions_df)
        
        # logging.info(f"closest_question: {closest_question}")
        
        # if closest_question:
        #     row = questions_df[questions_df['questions'] == closest_question].iloc[0]
        #     tags = row["tags"] if pd.notna(row["tags"]) else ""
        #     link = row["links"] if pd.notna(row["links"]) else None
        # else:
        #     tags = ""
        #     link = None

        
        response = user_input(user_question)
        logging.info(f"response: {response}")
        st.session_state.chat_history.append({'question': user_question, 'answer': response[0]})

        # Display the most recent interaction at the top
    if st.session_state.chat_history:
            # with st.container(): 
                latest_entry = st.session_state.chat_history[-1]
                st.write(f"**שאלה:** {latest_entry['question']}")
                if latest_entry.get('diagram'):
                    st.pyplot(latest_entry['diagram'])
                st.write(f"**תשובה:** {latest_entry['answer']}")
                st.write("---")  # Separator line
    
    # Display the rest of the chat history below
    with st.expander("ראה את ההיסטוריה המלאה"):
        for entry in reversed(st.session_state.chat_history[:-1]):
            st.write(f"**שאלה:** {entry['question']}")
            if entry.get('diagram'):
                st.pyplot(entry['diagram'])
            st.write(f"**תשובה:** {entry['answer']}")
            st.write("---")  # Separator line

if __name__ == "__main__":
    main()
