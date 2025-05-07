__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3
#All library and related imports.
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing import List
from typing import TypedDict
import streamlit as st


load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
tavily = os.environ["TAVILY_API_KEY"]



@st.cache_resource(show_spinner="Initializing RAG retriever...")
def createRAG(doc_path):
    loader = DirectoryLoader(doc_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()

    return vectorstore.as_retriever()
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})





# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "wiki_search","Tavily_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia ,a vectorstore or a internet search",
    )



llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)
llm_for_writer = ChatGroq(groq_api_key=groq_api_key, model_name = "llama-3.1-8b-instant")

# Prompt
system = """You are an expert at routing a user question to a vectorstore , wikipedia or a internet search via tavily.
The vectorstore contains documents related to the company Inflero , all related info and companies policies etc.
Use the vectorstore for questions on these topics.Use wiki-search for factual data, and whenever a concurrent question or news , use Tavily_search
Any question not related to company , like a news or any other info should strictly be redirected to wiki-search or Tavily search and not vectorstore"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

#setting up wikipedia wrapper 


api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

## Graph



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

from langchain.schema import Document

retriever = createRAG("Rag")

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print("fetched data through RAG : ",documents)
    return {"documents": documents, "question": question}

def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki.invoke({"query": question})
    #print(docs["summary"])
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)
    print("fetched data through wikipedia : ",wiki_results)

    return {"documents": wiki_results, "question": question}

import requests
import json

def TavSearch(state):
    """
    search the internet based on the re-phrased question via tavily api.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    question = state["question"]
    url = "https://api.tavily.com/search"
    payload = {
    "query": question,
    "topic": "general",
    "search_depth": "basic",
    "chunks_per_source": 3,
    "max_results": 1,
    "time_range": None,
    "days": 7,
    "include_answer": True,
    "include_raw_content": False,
    "include_images": False,
    "include_image_descriptions": False,
    "include_domains": [],
    "exclude_domains": []
    }
    
    headers = {
    "Authorization": f'Bearer {tavily}',
    "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    parsed = json.loads(response.text)
    content = parsed["answer"]
    search_results = Document(page_content=content)


    print("fetched data through tavily : ",search_results)

    return {"documents": search_results, "question": question}

def chatbot(state):
    """
    write final answer based on data fetched.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates generation key with final output
    """
    print("---Framing the Final answer via chatbot---")
    print("---Final answer---")
    question = state["question"]
    docs = state["documents"]
    prompt_for_writer = f"""
    You are a official RAG-Powered Multi-Agent Q&A Assistant for Inflera that has to answer the user's question in a consise yet ellaborate way , to only focus on key details that are essential while answering,
    you are final node of a agentic framework that is supplied the data as: {docs} from prev nodes, and based on this info and the question {question}, phrase a final crisp answer
    for the user, make sure to cover all technical terms and be concise.If the info supplied to you is inconsiderate to the question asked, you must NOT use your own learnings to reply sensibly but only if you are sure.
    Just straight ahead begin with your reply , nothing more or less. Begin with a greeting and answer.Make sure to not specify the provided documents part in your answer , just answer like a official assistant.
    """
    output = llm_for_writer.invoke(prompt_for_writer)
    state["generation"] = output.content
    return {"documents": docs, "question": question ,"generation" : output.content }

def route_question(state):
    """
    Route question to wiki search , RAG or tavily search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source.datasource == "Tavily_search":
        print("---ROUTE QUESTION TO tavily search---")
        return "Tavily_search"
    
from langgraph.graph import END, StateGraph, START


workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("wiki_search", wiki_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("SearchInternet", TavSearch)  # retrieve
workflow.add_node("chatbot", chatbot)  # Final writer



# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
        "Tavily_search": "SearchInternet"
    },
)
#workflow.add_edge( "retrieve", END)
#workflow.add_edge( "wiki_search", END)
workflow.add_edge( "wiki_search", "chatbot")
workflow.add_edge( "retrieve", "chatbot")
workflow.add_edge( "SearchInternet", "chatbot")
workflow.add_edge( "chatbot", END)
# Compile
app = workflow.compile()

def invoke(question):
  inputs = {"question": question}
  final_state = app.invoke(inputs)
  return (final_state["generation"])

#st.set_page_config(page_title="Finlera Multiagent", page_icon=":robot:", layout="wide", initial_sidebar_state="collapsed")




# App title
st.title("💬 Inflera RAG-Powered Multi-Agent Q&A")

# Text input area
user_input = st.text_area("Type your Question:", height=60)

# Submit button
if st.button("Submit"):
    if user_input.strip() != "":
        response = invoke(user_input)
        #st.markdown(f"**You:** {user_input}")
        st.markdown(response)
    else:
        st.warning("Please enter a message before submitting.")
