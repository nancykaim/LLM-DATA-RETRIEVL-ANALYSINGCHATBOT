import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering
from streamlit_chat import message
import tempfile

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="Client Data Survey System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load TAPAS Model
@st.cache_resource
def load_tapas_model():
    model_name = "google/tapas-large-finetuned-wtq"
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
    pipe = pipeline("table-question-answering", model=model, tokenizer=tokenizer)
    return pipe

pipe = load_tapas_model()

def get_answer(table, query):
    answers = pipe(table=table, query=query)
    return answers

def convert_answer(answer):
    if answer['aggregator'] == 'SUM':
        cells = answer['cells']
        converted = sum(float(value.replace(',', '')) for value in cells)
        return converted

    if answer['aggregator'] == 'AVERAGE':
        cells = answer['cells']
        values = [float(value.replace(',', '')) for value in cells]
        converted = sum(values) / len(values)
        return converted

    if answer['aggregator'] == 'COUNT':
        cells = answer['cells']
        converted = sum(int(value.replace(',', '')) for value in cells)
        return converted

    else:
        return answer['answer']

def get_converted_answer(table, query):
    converted_answer = convert_answer(get_answer(table, query))
    return converted_answer

# Initialize session state
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None

# Custom CSS for theme

# Web Application
st.title("TWINCY AI CHATBOT 📝")
st.write(' ')
st.markdown("⎛⎝ ≽  >  ⩊   < ≼ ⎠⎞")

# Sidebar Menu
with st.sidebar:
    st.title("Navigation")
    menu = st.radio("Go to", ("Upload Data", "Chat", "Analyze"))

if menu == "Upload Data":
    st.subheader("Upload your data for analysis ৻(  •̀ ᗜ •́  ৻)")
    uploaded_file = st.file_uploader("Upload your file here 🐻‍❄️ྀིྀི", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Load the data
        try:
            dataframe = pd.read_csv(temp_file_path)
            st.session_state.dataframe = dataframe
            st.session_state.uploaded = True
            st.session_state.chat_history = []  # Reset chat history on new upload
            st.write("**Data Preview:**")
            st.write(dataframe.head())
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.session_state.uploaded = False

if menu == "Chat" and st.session_state.uploaded:
    dataframe = st.session_state.dataframe

    # Convert all cells in the dataframe to strings
    dataframe = dataframe.astype(str)

    # Display chat history
    for idx, message_entry in enumerate(st.session_state.chat_history):
        if message_entry['role'] == 'user':
            message(message_entry['content'], is_user=True, key=f'{idx}_user', avatar_style="big-smile")
        else:
            message(message_entry['content'], key=f'{idx}_bot', avatar_style="thumbs")

    # User input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your question here...", key="input")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        if not user_input:
            st.write("Please enter a question.")
        else:
            # Append user query to chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})

            # Get and format the answer
            try:
                answer = get_converted_answer(dataframe, user_input)
                response = f"Answer: {answer}"
            except Exception as e:
                response = f"An error occurred: {e}"

            # Append bot response to chat history
            st.session_state.chat_history.append({'role': 'bot', 'content': response})

            # Refresh the page to show new messages
            st.experimental_rerun()
elif menu == "Analyze" and st.session_state.uploaded:
    dataframe = st.session_state.dataframe
    
    st.write("**Analyze Data (づ ᴗ _ᴗ)づ♡**")

    with st.form(key="analyze_form"):
        # Select columns
        columns = dataframe.columns.tolist()
        x_col = st.selectbox("Select X-axis column", columns)
        y_col = st.selectbox("Select Y-axis column", columns)
        
        # Select chart type
        chart_type = st.selectbox("Select chart type", ["Bar Chart", "Pie Chart", "Line Plot"])
        
        submit_button = st.form_submit_button(label="Generate Chart")

    if submit_button:
        try:
            fig, ax = plt.subplots()

            # Check if x-axis column is categorical
            if pd.api.types.is_numeric_dtype(dataframe[x_col]):
                # If x-axis is numeric, use it directly for line plot
                grouped_data = dataframe.groupby(x_col)[y_col].sum()
                if chart_type == "Line Plot":
                    grouped_data.plot(kind='line', ax=ax)
                elif chart_type == "Bar Chart":
                    grouped_data.plot(kind='bar', ax=ax)
                elif chart_type == "Pie Chart":
                    grouped_data.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            else:
                # Handle categorical x-axis
                if chart_type == "Line Plot":
                    # Aggregate y-values for each category
                    aggregated_data = dataframe.groupby(x_col)[y_col].sum()
                    aggregated_data.plot(kind='line', ax=ax)
                elif chart_type == "Bar Chart":
                    # Count occurrences of each category
                    grouped_data = dataframe.groupby(x_col).size()
                    grouped_data.plot(kind='bar', ax=ax)
                elif chart_type == "Pie Chart":
                    # Count occurrences of each category
                    grouped_data = dataframe.groupby(x_col).size()
                    grouped_data.plot(kind='pie', ax=ax, autopct='%1.1f%%')

            st.pyplot(fig)
        except Exception as e:
            st.write(f"An error occurred: {e}")
else:
    st.write("Please upload a CSV file to proceed.")
