# Databricks notebook source
# MAGIC %pip install pypdf
# MAGIC %pip install chromadb

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFacePipeline
from langchain.llms import HuggingFaceHub

# Manual Model building
from transformers import pipeline

# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Write PDF Texts to Delta Live Table") \
    .getOrCreate()

def read_the_pdf(file_name): 
    # As a first step we need to load and parse the document
    file_to_load = f'/Volumes/hackathon_data/documents/extreme_data/{file_name}.pdf' 

    loader = PyPDFLoader(file_to_load)
    # This splits it into pages
    pages = loader.load_and_split()

    # We will feed all pages in
    # chunk_size is a key parameter.
    # For more advanced use we may want to tune this or use a paragraph splitter or something else
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    text2doc= texts[:]
    doclist = []
    for i in range(len(texts)):
        doc = text2doc[i]
        doclist.append(doc.page_content)

    return doclist


data = []
# Loop through PDF files and add extracted texts to table contents
id = 12340
for item in dbutils.fs.ls('/Volumes/hackathon_data/documents/extreme_data'):
    if item.name.endswith('.pdf'):
        
        file_name = item.name[:-4]  # Remove the '.pdf' extension from the file name
        file_path = item.path
        print(file_name)
        print(file_path)
        texts_cont = read_the_pdf(file_name)

        for item in texts_cont:
            # add_to_table_contents(texts_cont, file_name, file_path)
            # Append data to the list
            id+=1 
            data.append((id, file_name, file_path, item))

# Create DataFrame from the collected data
df_files = spark.createDataFrame(data, ["id", "file_name", "file_path", "contents"])

df_files.write.format("delta").saveAsTable("hackathon_data.documents.PDF_data_table1")

# Show the DataFrame
df_files.show()
