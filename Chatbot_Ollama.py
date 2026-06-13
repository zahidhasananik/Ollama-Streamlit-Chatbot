import streamlit as st
import os
# Core components from langchain_core:
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler 

# Community integrations (Ollama এবং OpenAI/Groq এর জন্য আলাদা ইমপোর্ট)
from langchain_community.chat_models import ChatOllama 
from langchain_openai import ChatOpenAI

# --- 1. CONFIGURATION & ENVIRONMENT CHECK ---
OLLAMA_MODEL = 'llama3:8b' # লোকাল মেশিনের জন্য
CLOUD_MODEL = 'llama3-8b-8192' # ক্লাউড (Groq) এর জন্য
ASSISTANT_NAME = "ZAHID"

# চেক করা হচ্ছে অ্যাপটি ক্লাউডে (Hugging Face) আছে নাকি আপনার নিজের পিসিতে আছে
is_hf_cloud = "GROQ_API_KEY" in os.environ

# System instruction for the prompt template
SYSTEM_INSTRUCTION = (
    f"You are a helpful and expert programming assistant named '{ASSISTANT_NAME}'. "
    "You must respond in a formal, teaching tone. "
    "Always use Python code examples in your answers, formatted with Markdown."
)

# Define the Prompt Template structure
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("placeholder", "{history}"),
    ("human", "{user_input}"),
])

# --- 2. STREAMLIT SESSION STATE SETUP (MEMORY) ---

# Initialize chat history for display
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": f"Hello! I am {ASSISTANT_NAME}, powered by a hybrid LLM setup. How can I assist with your questions?"}
    ]

# Initialize the model connector (cached for performance)
@st.cache_resource
def get_llm_model():
    if is_hf_cloud:
        # ক্লাউড সেটআপ: Groq API এবং LangChain-এর ChatOpenAI ক্লায়েন্ট ব্যবহার হবে
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            return ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                model=CLOUD_MODEL,
                streaming=True
            )
        except Exception as e:
            st.error(f"Error connecting to Cloud Provider (Groq): {e}")
            st.stop()
    else:
        # লোকাল সেটআপ: আপনার পিসির লোকাল Ollama ব্যবহার করা হবে
        try:
            return ChatOllama(model=OLLAMA_MODEL, streaming=True)
        except Exception as e:
            st.error(f"Error connecting to Ollama: {e}. Ensure Ollama is running and model '{OLLAMA_MODEL}' is pulled.")
            st.stop()

# --- 3. DISPLAY THE CHAT INTERFACE ---

st.set_page_config(page_title=f"{ASSISTANT_NAME} Chatbot")
st.title(f"🤖 {ASSISTANT_NAME} Intelligent Chatbot")

# লাইভ স্ট্যাটাস ইন্ডিকেটর
if is_hf_cloud:
    st.caption(f"⚡ Status: Running on Cloud via Groq ({CLOUD_MODEL})")
else:
    st.caption(f"💻 Status: Running on Local Machine via Ollama ({OLLAMA_MODEL})")

# Display all user-visible messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. HANDLE NEW USER INPUT ---

if prompt := st.chat_input("Ask Zahid a question..."):
    # 4a. Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4b. Prepare history for the prompt
    history = []
    for msg in st.session_state.messages[1:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    # 4c. Generate AI Response (with streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare the final prompt payload
        formatted_prompt = prompt_template.format_messages(
            history=history,
            user_input=prompt
        )
        
        # Call the appropriate model and stream the response
        try:
            llm = get_llm_model()
            
            # Handles the streaming output from the model (Ollama/Groq দুইটার জন্যই লুপ এক থাকবে)
            for chunk in llm.stream(formatted_prompt):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌") # Use ▌ for typing indicator

            # Final response without the typing indicator
            message_placeholder.markdown(full_response)
            
            # 4d. Add final response to Streamlit history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            if is_hf_cloud:
                st.error(f"**[Error]** Cloud model failed to respond. Details: {e}")
            else:
                st.error(f"**[Error]** Failed to communicate with Ollama server. Is Ollama running? Details: {e}")

# --- 5. OPTIONAL: CLEAR HISTORY ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.rerun()on_state.messages[0]]
    st.rerun()
