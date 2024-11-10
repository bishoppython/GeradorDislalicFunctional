import chromadb
import pandas as pd
import json
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import re  # Para manipulação de strings

# Configuração do modelo e prompt
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.4)
prompt_template = """
Você é um assistente gerador de datasets sobre dislalia. Classifique as palavras com as seguintes regras:
Omissão: quando existe alguma letra da palavra que não está presente, mudando o sentido da palavra ou a deixando incompreensível.
Substituição: ocorre quando alguma palavra ou letra é trocada por outra, alterando sua pronúncia.
Acréscimo: ocorre quando uma letra é acrescentada, mudando o sentido da palavra.
Responda em formato JSON no seguinte padrão:
{{ "input": "{input}", "correction": "{correction}", "words": [...], "labels": [...] }}
"""

# Inicializando o ChromaDB Client
client = chromadb.Client()
collection_name = "dados_dislia"

# Verifica se a coleção já existe; se não, cria e adiciona dados
try:
    collection = client.get_collection(collection_name)
except:
    collection = client.create_collection(name=collection_name)
    # Processamento do arquivo XLSX e armazenamento no ChromaDB
    xlsx_path = "DISLALIC_DATASET_RAG.xlsx"
    df = pd.read_excel(xlsx_path)
    for idx, row in df.iterrows():
        # Adicionando cada linha como um documento no ChromaDB
        collection.add({"input": row['input'], "correction": row['correction']})

# Função de geração de respostas com RAG       
def generate_response_with_rag(user_input):
    # Realiza a consulta no ChromaDB para obter contextos relevantes
    results = collection.query(query_texts=[user_input], n_results=3)
    context = "\n".join([doc["correction"] for doc in results if "correction" in doc])

    # Configura o prompt com o contexto e o input do usuário
    prompt = PromptTemplate.from_template(prompt_template)
    final_prompt = prompt.format(input=user_input, correction=context)

    # Geração de resposta usando o modelo
    response = llm(final_prompt)

    # Verificação e depuração da resposta antes de decodificar como JSON
    if response:
        print("Resposta do modelo:", response)  # Depuração: exibir resposta
        
        # Remover crases e espaços em branco adicionais, se houver
        cleaned_response = re.sub(r"```(json)?", "", response).strip()

        try:
            response_json = json.loads(cleaned_response)  # Tentar carregar como JSON
            # Verifica se a resposta contém dislalia
            if "words" in response_json and response_json["words"]:
                return response_json
            else:
                return {"message": "Este conteúdo não possui ou não foi identificado traços de dislalia funcional"}
        except json.JSONDecodeError:
            print("Erro: A resposta não está em formato JSON válido.")
            return {"error": "A resposta do modelo não está em JSON válido."}
    else:
        print("Erro: A resposta do modelo está vazia.")
        return {"error": "A resposta do modelo está vazia."}

# Configuração da Página
st.set_page_config(page_title="TEAChat - Dislalia Assistant", page_icon="ico/heart.png")
st.title("TEAChat - Dislalia Assistant")

# Entrada do Usuário e Exibição de Respostas
if "messages" not in st.session_state:
    st.session_state.messages = []

if user_input := st.chat_input("Insira a frase a ser corrigida:"):
    response = generate_response_with_rag(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Exibe a resposta
    st.json(response)
