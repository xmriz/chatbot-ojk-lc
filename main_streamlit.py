import streamlit as st
from utils.config import get_config_streamlit
from utils.vector_store import PineconeIndexManager
from utils.model_config import ModelName, get_model
from utils.rag_chain_with_chat_history import create_chain_with_chat_history
from dotenv import load_dotenv
from utils.chat_history import ChatHistory
import hmac

# Apply asyncio and load environment variables
load_dotenv()

config = get_config_streamlit()

# Templates for prompts
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question WITH ITS ORIGINAL LANGUAGE. if the follow \
up question is not clear. 
If the chat history is not relevant to the follow up question, please ignore it.

Chat history:
{chat_history}
Follow up question: {question}
Standalone question:"""

_ANSWER_TEMPLATE = """Context information is below.
context: {context}

Given the context and the metadata information and not prior knowledge, \
answer the query asking about banking compliance in Indonesia. 
Answer the question based on the context and the metadata information.
ALWAYS ANSWER WITH USER'S LANGUAGE.
Please provide your answer with [regulation_number](file_url) in metadata \
(if possible) in the following format:

Answer... \n\n
Source: [metadata['regulation_number']](metadata['file_url'])

But if you cannot find the regulation number, just provide the answer.

Question: {question}
"""


# Page configuration
st.set_page_config(page_title="OJK Chatbot", page_icon="🤖", layout="centered")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK"}
    ]

TOP_K = 3
model_name = ModelName.AZURE_OPENAI


def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", help="Input the password to continue",
                  on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


@st.cache_resource(show_spinner=False)
def load_chain():
    TOP_K = 6
    llm_model, embed_model = get_model(
        model_name=ModelName.OPENAI, config=config)

    pinecone = PineconeIndexManager(index_name='ojk', embed_model=embed_model, config=config)
    vector_store = pinecone.load_vector_index()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K})
    chain = create_chain_with_chat_history(
        contextualize_q_prompt_str=_TEMPLATE,
        qa_system_prompt_str=_ANSWER_TEMPLATE,
        retriever=retriever,
        llm_model=llm_model,
    )
    return chain


# Initialize chain and chat history
if "chain" not in st.session_state:
    st.session_state.chain = load_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()

if not check_password():
    st.stop()

st.title("Chat with the OJK BOT 💬🤖")

# User input handling
if prompt := st.chat_input("Ask me a question about any Banking Compliance in Indonesia"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Thinking..."):
            answer_chunks = []
            for chunk in st.session_state.chain.stream(
                {"chat_history": st.session_state.chat_history, "question": prompt}
            ):
                if "question" in chunk:
                    question = chunk["question"]
                if "answer" in chunk:
                    answer_chunks.append(chunk["answer"])
                    # Update the placeholder with the current answer
                    placeholder.write("".join(answer_chunks))

            answer = "".join(answer_chunks)
            st.session_state.chat_history.add_chat(prompt, answer)
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)
            print(question)

# Reset button
if st.button("Reset"):
    st.session_state.chat_history.clear_history()
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK"
        }
    ]
    st.rerun()
