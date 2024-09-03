
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


# Define the custom FancyBboxPatch and Axes classes
class StaticColorAxisBBox(mpatches.FancyBboxPatch):
    def set_edgecolor(self, color):
        if hasattr(self, "_original_edgecolor"):
            return
        self._original_edgecolor = color
        self._set_edgecolor(color)

    def set_linewidth(self, w):
        super().set_linewidth(1)

class FancyAxes(maxes.Axes):
    name = "fancy_box_axes"
    _edgecolor: str

    def __init__(self, *args, **kwargs):
        self._edgecolor = kwargs.pop("edgecolor", None)
        super().__init__(*args, **kwargs)

    def _gen_axes_patch(self):
        return StaticColorAxisBBox(
            (0, 0),
            1.0,
            1.0,
            boxstyle="round, rounding_size=0.06, pad=0",
            edgecolor=self._edgecolor,
            linewidth=1,
        )


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
            logging.info(f"full response : {response}")
            
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
                        fig, ax = plt.subplots()
                        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
                        bars = ax.bar(categories, values, label=categories, color=bar_colors)
                        ax.set_ylim(0, max(values) * 1.2)
                        plt.xticks(rotation=45)
                        # Add value labels on top of the bars with a small font size
                        # if len(values) > 1:
                        #     for bar in bars:
                        #         yval = bar.get_height()
                        #         ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval}', ha='center', va='bottom', fontsize=8)
                        # ax.legend()


                        # Register the custom projection
                        register_projection(FancyAxes)
                        
                        # Replace the original part with this
                        fig = plt.figure()
                        ax = fig.add_subplot(
                            111, projection="fancy_box_axes", facecolor="white", edgecolor="black"
                        )
                        ax.spines[["bottom", "left", "right", "top"]].set_visible(False)
                        
                        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
                        bars = ax.bar(categories, values, label=categories, color=bar_colors)
                        ax.set_ylim(0, max(values) * 1.2)
                        plt.xticks(rotation=45)

                        
                        # Add value labels on top of the bars with a small font size
                        if len(values) > 1:
                            for bar in bars:
                                yval = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval}', ha='center', va='bottom', fontsize=8)
                        
                        # frame = legend.get_frame()
                        # frame.set_boxstyle('round,pad=0.5,rounding_size=0.2')
                        ax.legend()
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
    # Load the vector store and perform a similarity search
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Use the content of the documents to form a context
    context = " ".join([doc.page_content for doc in docs])
   
    # Combine the context with the user question and generate a response
    prompt = f"הקשר: {context}\nשאלה: {user_question}\nתשובה:"
    logging.info(f"prompt: {prompt}")
    
    # response, diagram = generate_response(prompt, row["diagram"])
    response, diagram = generate_response(prompt, diagram_data)
    # st.write(response)
    logging.info(f"response, diagram: {response, diagram}")
    return  response, diagram


def parse_diagram_data(diagram_str):
    # Extract categories and values using regular expressions
    categories_part = re.search(r'categories = \[(.*?)\]', diagram_str).group(1)
    values_part = re.search(r'values = \[(.*?)\]', diagram_str).group(1)

    # Convert the strings to lists
    categories = categories_part.split(',')
    # logging.info(f"categories: {categories}")
    values = list(map(int, values_part.split(',')))
    return categories, values

# Define a function to reset the inputs
def reset_inputs():
    st.session_state.question_key += 1
    st.session_state.select_key += 1

def reset_conversation():
    st.session_state.chat_history = []
    
def main():

    st.set_page_config("Chat PDF")
    # st.set_page_config(layout="centered", initial_sidebar_state="collapsed")
    # st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="Chat PDF", page_icon="🤖")

    
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

    
    questions_df = load_questions('data/knowledge_center.csv')
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
                response = user_input(user_question)  # Generate the response
                st.session_state.chat_history.append({'question': user_question, 'answer': response[0]})
                st.session_state['last_processed'] = user_question  # Track last processed question

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
    
    # Load the vector store (initialization, not directly related to user interaction)
    with st.spinner("טוען נתונים..."):
        raw_text = get_pdf_text(PDF_FILE_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
   

if __name__ == "__main__":
    main()
