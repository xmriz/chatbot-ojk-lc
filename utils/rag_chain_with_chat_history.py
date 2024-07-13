import json
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough

from utils.chat_history import ChatHistory

from langchain_core.retrievers import BaseRetriever

from utils.final_chain_input_type import InputTypeFinalChain

from utils.model_config import ModelName

# ===== formatting functions =====


def _format_metadata(metadata):
    """Remove filename from metadata."""
    # check if file_name is in metadata
    if "file_name" in metadata:
        metadata.pop("file_name", None)
    return metadata


def _combine_documents(docs):
    """Combine documents into a single JSON string."""
    doc_list = [{"metadata": _format_metadata(doc.metadata), "page_content": doc.page_content} for doc in docs]
    return json.dumps(doc_list, indent=2)


def create_chain_with_chat_history(contextualize_q_prompt_str: str, qa_system_prompt_str: str, retriever: BaseRetriever, llm_model: ModelName):
    CONTEXTUALIZE_Q_PROMPT_STR = contextualize_q_prompt_str
    QA_SYSTEM_PROMPT_STR = qa_system_prompt_str
    QA_PROMPT = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT_STR)
    CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(CONTEXTUALIZE_Q_PROMPT_STR)
    _inputs_question = RunnableMap(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: x["chat_history"].get_formatted_history()
        )
        | CONTEXTUALIZE_Q_PROMPT
        | llm_model
        | StrOutputParser(),
    )
    _context_chain = {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": lambda x: x["standalone_question"],
    }
    conversational_qa_with_context_chain = (
        _inputs_question
        | _context_chain
        | {
            "question": itemgetter("question"),
            "answer": QA_PROMPT | llm_model | StrOutputParser(),
            "context": itemgetter("context"),
        }
        # | StrOutputParser()
    )
    final_chain = conversational_qa_with_context_chain.with_types(input_type=InputTypeFinalChain)
    return final_chain


def get_response(chat_history: ChatHistory, question: str, chain):
    response = chain.invoke({"chat_history": chat_history, "question": question})
    chat_history.add_chat(response["question"], response["answer"])
    return response



async def print_answer_stream(chat_history: ChatHistory, question: str, chain):
    answer_chunks = []
    async for chunk in chain.astream({"chat_history": chat_history, "question": question}):
        if 'question' in chunk:
            question = chunk['question']
        if 'answer' in chunk:
            answer_chunks.append(chunk['answer'])
            print(chunk['answer'], end='', flush=True)

    answer = ''.join(answer_chunks)
    chat_history.add_chat(question, answer)


# # ===== Prompt Templates =====

# CONTEXTUALIZE_Q_PROMPT_STR = """Given the following conversation and a follow up question, rephrase the 
# follow up question to be a standalone question WITH ITS ORIGINAL LANGUAGE. if the follow \
# up question is not clear.

# Chat history:
# {chat_history}
# Follow up question: {question}
# Standalone question:"""

# CONTEXTUALIZE_Q_PROMPT = PromptTemplate.from_template(CONTEXTUALIZE_Q_PROMPT_STR)

# # QA_PROMPT
# QA_SYSTEM_PROMPT_STR= """Context information is below.
# context: {context}

# Given the context and the metadata information and not prior knowledge, \
# answer the query asking about banking compliance in Indonesia. 
# Answer the question based on the context and the metadata information.
# ALWAYS ANSWER WITH USER'S LANGUAGE.
# Please provide your answer with [regulation_number](file_url) in metadata 
# (if possible) in the following format:

# Answer... \n\n
# Source: [metadata['regulation_number']](metadata['file_url'])

# Question: {question}
# """
# QA_PROMPT = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT_STR)


# # ===== Chaining =====

# # generate question from chat history if needed
# _inputs_question = RunnableMap(
#     standalone_question=RunnablePassthrough.assign(
#         chat_history=lambda x: _format_chat_history(x["chat_history"])
#     )
#     | CONTEXTUALIZE_Q_PROMPT
#     | llm_model
#     | StrOutputParser(),
# )

# _context_chain = {
#     "context": itemgetter("standalone_question") | retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }

# conversational_qa_with_context_chain = (
#     _inputs_question
#     | _context_chain
#     | {
#         "question": itemgetter("question"),
#         "answer": QA_PROMPT | llm_model | StrOutputParser(),
#         "context": itemgetter("context"),
#     }
#     # | StrOutputParser()
# )

# final_chain = conversational_qa_with_context_chain.with_types(input_type=InputTypeFinalChain)

