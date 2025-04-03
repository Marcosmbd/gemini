import streamlit as st
from google import genai
from google.genai import types
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import json
import os

# Recupera a chave de serviço do secrets.toml
service_key_json = st.secrets["key"]["service_key"]

# Converte o JSON para um dicionário
service_key_dict = json.loads(service_key_json)

# Salva a chave de serviço em um arquivo temporário
service_key_path = "gcp_service_key.json"

with open(service_key_path, "w") as f:
    json.dump(service_key_dict, f)

# Configura a variável de ambiente
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_path

try:
    credentials, project = default()
except DefaultCredentialsError:
    st.error("Erro: Credenciais não encontradas. Verifique se configurou corretamente.")
    st.stop()

def generate(instruction, prompt):
    client = genai.Client(
        vertexai=True,
        project="marcos-estudos",
        location="global",
    )

    model = "gemini-2.0-flash-001"
    
    # Construindo o contexto da conversa a partir do histórico
    contents = []
    for message in st.session_state.messages:
        role = "user" if message["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=message["content"])]))

    # Adiciona a pergunta atual
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    tools = [
        types.Tool(retrieval=types.Retrieval(vertex_ai_search=types.VertexAISearch(
            datastore=st.secrets["google_cloud"]["datastore"]
        )))]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        tools=tools,
        system_instruction=[types.Part.from_text(text=instruction)],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,  # 🔹 Agora o Gemini recebe TODO o histórico
        config=generate_content_config,
    )

    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text
    else:
        return "Resposta não encontrada."

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        .footer {
            position: fixed;
            bottom: 10px;
            left: 10px;
            font-size: 12px;
            color: gray;
            opacity: 0.7;
            z-index: 9999;
        }
    </style>
    <div class="footer">By Marcos B.</div>
""", unsafe_allow_html=True)

# --- Inicializar histórico do chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Título do app ---
st.title("Pergunte ao Gemini 🤖")

# --- Caixa de texto fixa para a instrução (acima) ---
st.session_state.instruction = st.text_area(
    "Dê instruções ao Gemini (opicional)", 
    value=st.session_state.get("instruction", "Forneça respostas diretas e objetivas."),
    height=100
)

# --- Exibir mensagens do histórico ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Campo para o prompt (abaixo) ---
prompt = st.chat_input("Digite sua pergunta")

if prompt:
    # Adicionar mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gerar resposta da IA
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            ai_response = generate(st.session_state.instruction, prompt)  # Pegando a instrução salva
            full_response = ai_response
        except Exception as e:
            full_response = f"Ocorreu um erro ao gerar a resposta: {e}"
            st.error(full_response)

        # Exibir resposta e atualizar histórico
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
