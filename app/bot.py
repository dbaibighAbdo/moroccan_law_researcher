import streamlit as st
from helpers.utils import write_message
from agent import generate_response
from langchain_core.messages import SystemMessage, HumanMessage
import uuid
from arabic_support import support_arabic_text

# Support Arabic text alignment in all components
support_arabic_text(all=True)
# Page Config
st.set_page_config("chatbot juridique", page_icon=":robot_face:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحبًا! أنا مساعدك القانوني المتخصص في القانون المغربي. كيف يمكنني مساعدتك اليوم؟"},
    ]
import uuid

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

def handle_submit(message):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.spinner('جاري التفكير...'):
        response = generate_response(message, config)
        # Optionally save to session_state.messages for display
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)