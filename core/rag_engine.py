#Actionableitems , decision , questions 

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, get_retriever
import os 


def get_llm():
    return ChatGroq(model = "openai/gpt-oss-120b", groq_api_key = os.getenv("GROQ_API_KEY"),temperature=0.2)



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store)

    return (
        {
            "context" : RunnableLambda(lambda x : retriever.invoke(x["question"]))
                        | RunnableLambda(lambda docs : "\n\n".join(d.page_content for d in docs)),
            "question" : RunnableLambda(lambda x : x["question"]),
        }
        | ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering questions about a meeting. "
             "Use only the following context to answer. If the answer is not in the "
             "context, say 'I could not find that in the meeting.'\n\n"
             "Context:\n{context}"),
            ("human","{question}"),
        ])
        | get_llm()
        | StrOutputParser()
    )


def ask_question(rag_chain, question: str) -> str:
    return rag_chain.invoke({"question" : question})


def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)