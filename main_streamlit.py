# import nest_asyncio
import streamlit as st
from utils.vector_store import PineconeIndexManager
from utils.model_config import ModelName, get_model
from utils.rag_chain_with_chat_history import create_chain_with_chat_history
from dotenv import load_dotenv
import hmac
from utils.config import get_config_streamlit
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from utils.chat_history import ChatStore


# Apply asyncio and load environment variables
# nest_asyncio.apply()
load_dotenv()

# ============================== FUNCTIONS ==============================


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
def load_chain(config: dict = None, top_k: int = 10, top_n: int = 6, model_name: ModelName = ModelName.AZURE_OPENAI, prompt: str = None, _chat_store: ChatStore = ChatStore()):

    llm_model, embed_model = get_model(model_name=model_name, config=config)

    pinecone = PineconeIndexManager(
        index_name='ojk', embed_model=embed_model, config=config)
    vector_store = pinecone.load_vector_index()
    compressor = CohereRerank(
        cohere_api_key=config['cohere_api_key'], top_n=top_n)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}),
    )

    chain = create_chain_with_chat_history(
        prompt_str=prompt,
        get_session_history= _chat_store.get_session_history,
        retriever=retriever,
        llm_model=llm_model,
    )
    return chain


# ============================== PROMPTING ==============================
PROMPT = """The context information is below.
Context: 
{context}

Based on the context and the metadata information provided, \
answer the query related to banking compliance in Indonesia.
Use the context and metadata information only, WITHOUT RELYING ON EXTERNAL KNOWLEDGE.
ALWAYS ANSWER IN THE USER'S LANGUAGE.

Please provide your answer in the following markdown format, \
including the regulation number and file URL from metadata if available:

'''
[Your answer here] \n\n
Source: [metadata['regulation_number']](metadata['file_url'])
'''

If you cannot find the regulation number, just provide the answer without the source. 
If the file_url ends with '.pdf', you can add the metadata['page_number'] in the URL like this: 

'''
[Your answer here] \n\n
Source: [metadata['regulation_number']](metadata['file_url']#page=metadata['page_number'])
'''

DO NOT PROVIDE AMBIGUOUS ANSWERS.
IF THE QUESTION IS NOT RELEVANT TO THE CONTEXT, PLEASE SKIP THE QUESTION.
"""

# ============================== CONSTANTS ==============================
TOP_K = 10
TOP_N = 6
K_STORE = 3
model_name = ModelName.AZURE_OPENAI

# ============================== MAIN ==============================

config = get_config_streamlit()

# Page configuration
st.set_page_config(page_title="OJK Chatbot", page_icon="🤖", layout="centered")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK"}
    ]


# Initialize state variables
if "user_id" not in st.session_state:
    st.session_state.user_id = "xmriz"

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "ojk-conversation"

if "chat_store" not in st.session_state:
    st.session_state.chat_store = ChatStore(k=K_STORE)

if "chain" not in st.session_state:
    st.session_state.chain = load_chain(config=config, top_k=TOP_K, top_n=TOP_N, model_name=model_name, prompt=PROMPT, _chat_store=st.session_state.chat_store)


# Password check
if not check_password():
    st.stop()

# Page title
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

        with st.spinner("Generating response..."):
            answer_chunks = []
            for chunk in st.session_state.chain.stream(
                {"question": prompt},
                config={
                    "configurable": {
                        "user_id": st.session_state.user_id,
                        "conversation_id": st.session_state.conversation_id,
                    }
                }
            ):
                if "answer" in chunk:
                    answer_chunks.append(chunk["answer"])
                    placeholder.write("".join(answer_chunks))

            answer = "".join(answer_chunks)
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)

# Reset button
if st.button("Reset"):
    st.session_state.chat_store.clear_session_history(user_id=st.session_state.user_id, conversation_id=st.session_state.conversation_id)
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about any Regulation of BI and OJK"
        }
    ]
    st.rerun()
