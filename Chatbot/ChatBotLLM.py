# Databricks notebook source
# MAGIC %pip install flask-sqlalchemy 
# MAGIC %pip install databricks-genai-inference 
# MAGIC %pip install gradio

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow.deployments
from langchain_community.chat_models import ChatDatabricks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from databricks_genai_inference import ChatSession
import gradio as gr

# COMMAND ----------

deploy_client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

question = "What are the safety risks with floods?"

# COMMAND ----------

vsc = VectorSearchClient()
results = vsc.get_index('sw_db_hackathon', 'hackathon_data.documents.index_table').similarity_search(
  query_text=question,
  columns=["file_path", "contents"],
  num_results=1)

# COMMAND ----------

docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

import os

# COMMAND ----------

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

print(host)

# COMMAND ----------

token = 'include your db token'

# COMMAND ----------

os.environ["DATABRICKS_HOST"] = host
#Get the vector search index
vsc = VectorSearchClient(workspace_url=host, personal_access_token=token)
vs_index = vsc.get_index(
    endpoint_name='sw_db_hackathon',
    index_name='hackathon_data.documents.index_table')

# COMMAND ----------

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=token)
    vs_index = vsc.get_index(
        endpoint_name='sw_db_hackathon',
        index_name='hackathon_data.documents.index_table'
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="contents", embedding=embedding_model
    )
    return vectorstore.as_retriever()

# COMMAND ----------

chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 1000)
# print(f"Test chat model: {chat_model.predict('What is Apache Spark')}")

# COMMAND ----------

# embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
# TEMPLATE = """You are a helpful chatbot that is able to warn field workers of harmful weather incidents. You will use the historic data in table accuweather.forecast.us_postal_daynight_imperial to assist in your answer. Please include the safety guidelines for each of the weather warning from the documents provided. When answering the questions, please do not refer to the source paths.
# Use the following pieces of context to answer the question at the end:{context}

# Question: {question}

# Answer:
# """
# prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# chain = RetrievalQA.from_chain_type(
#     llm=chat_model,
#     chain_type="stuff",
#     retriever=get_retriever(),
#     chain_type_kwargs={"prompt": prompt}
# )


# COMMAND ----------

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
TEMPLATE = """You are a helpful chatbot that is able to warn field workers of harmful weather incidents. You will use the data in table accuweather.forecast.us_postal_daynight_imperial to assist in your answer. Include the short_phrase value from the table accuweather.forecast.us_postal_daynight_imperial in the answer as Weather Status. Please include the US national safety guidelines from hackathon_data.documents.pdf_data_table1 and vecor search data. do not mention the table name.

Use the following pieces of context to answer the question at the end:{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)


# COMMAND ----------

context = {"Historic weather data from the accuweather.forecast.us_postal_daynight_imperial table will be used to provide safety instructions for field workers."}

# COMMAND ----------

question = {"query": "I am a contractor undertaking excavation works near postal code 75840 on 2023-04-20 night time. Are the weather conditions safe for me to work?"}
# context = {"You are a weather agent who looks at work policies for working in extreme weather conditions."}
answer = chain.run(question)
print(answer)

# COMMAND ----------

def dbrx_chatbot(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = chain.run(inp)
    history.append((input, output))
    return history, history

# COMMAND ----------

block = gr.Blocks()


with block:
    gr.Markdown("""<h1><center>LLM Weather Agent</center></h1>
    """)
    chatbot = gr.Chatbot()
    message = gr.Textbox()
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(dbrx_chatbot, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug = True,share=True)
