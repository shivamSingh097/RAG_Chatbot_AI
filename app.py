import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType
from transformers import pipeline
from langchain_community.tools import SerperDevTool
from langchain_community import tools

load_dotenv()

DB_PATH = "chroma_db/"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_name = "sentence-transformers/all-MiniLM-L6-V2"
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

text_generation_pipeline = pipeline(
    "text-generation",
    model="distilbert/distilbert-base-uncased",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=vector_db.as_retriever(search_kwargs={"k":3})
# --- New Agent-based-logic ---
search_tool = tools.SerperDevTool()
tools = [search_tool]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

st.title("INDI_Powered_Chatbot ('Vocal for Local')")
st.write("Ask me anything about AI, Python, World Economic!")

user_question = st.text_input("Your Question:")

if user_question:
    st.write(f"Thinking about your question: {user_question}")
    local_answer = qa_chain({"query":user_question})
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_question)
    if docs:
        context = "\n\n".join([doc.page_content for doc in docs])
        st.write("---")
        st.markdown(f"**Answer from Local Knowledge Base:**")

        prompt_template = """
        You are a helpful assistant.
        Use the following context to answer the question.
        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context, question=user_question)
        response = llm.invoke(final_prompt)

        st.write("---")
        st.markdown(f"**Answer:** {response}")
        st.write("---")
        st.markdown(f"**Source Context:**")
        st.code(context)

    else:
        st.write("---")
        st.markdown(f"**Answer from Web Search:**")
        try:
            response = agent.run(user_question)
            st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred:{e}.The model might be struggling to process the output from the search tools")

# "prompt_template = """
# You are a helpful assistant. Use the following context to answer the question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Context: {context}
# Question: {question}
# Answer:
# """

# prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

# st.title("INDI_Powered_Chatbot ('Vocal for Local')")
# st.write("Ask me anything about AI, Python, World Economic!")

# user_question = st.text_input(Your Question):

# if user_question:
#    st.write(f"Thinking about your question: {user_question}")
#   docs = retriever.invoke(user_question)
#   context = "\n\n".join([doc.page_content for doc in docs])
#   final_prompt = prompt.format = prompt.format(context=context, question=user_question)

#   response = llm.invoke(final_prompt)
#
#    st.write("---")
#    st.markdown(f"**Answer:** {response}")
#    st.write("---")
#     st.markdown(f"**Source Context:**")
#     st.code(context)
