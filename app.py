import streamlit as st
import sqlite3
import hashlib
import json
from typing import Dict, List, Literal
from typing_extensions import TypedDict, Annotated
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any, Annotated, Tuple, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage, AnyMessage
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from typing import Annotated
import json
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import display, Image
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import SkipValidation, BaseModel
from langgraph.managed import IsLastStep, RemainingSteps
from dotenv import load_dotenv
load_dotenv()


# Configuración de la página
st.set_page_config(
    page_title="Sistema de Terapia Psicológica",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#----------------------------------
# CONFIGURACIÓN API GOOGLE
#----------------------------------

import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Configurar la API de Google Gemini correctamente
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Configuración de LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7,
    convert_system_message_to_human=False
)

#----------------------------------
# INICIALIZACIÓN DE ESTADO
#----------------------------------
class agentState(TypedDict):
  messages: Annotated[list[AnyMessage], add_messages]


#----------------------------------
# HERRAMIENTAS
#----------------------------------
DB_PATH ="psychology_system.db"

def get_db_connection():
    """Obtiene una conexión a la base de datos"""
    return sqlite3.connect(DB_PATH)

from datetime import datetime
@tool
def obtener_pregunta_por_id(pregunta_id: int) -> Optional[Dict[str, str]]:
    """Devuelve todos los campos de una pregunta dado su ID"""
    try:
        DB_PATH = "psychology_system.db"
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT id, categoria, pregunta, condicional FROM preguntas WHERE id = ?", (pregunta_id,))
        resultado = cursor.fetchone()

        conn.close()

        if resultado:
            return {
                "id": resultado[0],
                "categoria": resultado[1],
                "pregunta": resultado[2],
                "condicional": resultado[3]
            }
        else:
            return {
                "error": f"⚠️ No se encontró una pregunta con ID {pregunta_id}"
            }
    except Exception as e:
        return {
            "error": f"❌ Error al consultar la base de datos: {str(e)}"
        }

@tool
def psychological_diagnosis(caso_clinico: str) -> str:
    """Genera un diagnostico en base a un caso clínico"""
    prompt = f"""Eres un psicólogo clínico especializado en generar diagnósticos. Proporciona:
    1. Diagnóstico preliminar (usa formato DSM-5)
    2. 3 posibles diagnósticos diferenciales
    3. Recomendaciones iniciales

    Caso:
    {caso_clinico}

    Respuesta:"""

    return llm.invoke(prompt).content.strip()

from datetime import datetime

@tool
def save_patient_evaluation(
    paciente_id: int,
    nombre: str,
    telefono: str,
    correo: str,
    historia_clinica: str
) -> Dict[str, Any]:
    """
    Herramienta para que el agente guarde la evaluación completa del paciente.

    Args:
        paciente_id: cédula de identidad
        nombre: Nombre completo del paciente
        telefono: Número de teléfono del paciente
        correo: Correo electrónico del paciente
        historia_clinica: Historia clínica generada por el agente

    Returns:
        Dict con el resultado de la operación
    """
    try:
        # Guardar en la base de datos
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Insertar la evaluación
            cursor.execute('''
                INSERT INTO evaluaciones
                (paciente_id, nombre, telefono, correo, historia_clinica)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                paciente_id,
                nombre.strip(),
                telefono.strip() if telefono else "",
                correo.strip() if correo else "",
                historia_clinica.strip()
            ))

            evaluation_id = cursor.lastrowid
            conn.commit()

            return {
                "success": True,
                "evaluation_id": evaluation_id,
                "paciente_id": paciente_id,
                "message": f"Evaluación guardada exitosamente con ID {evaluation_id}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

    except sqlite3.Error as e:
        return {
            "success": False,
            "error": f"Error de base de datos: {str(e)}",
            "message": "No se pudo guardar la evaluación en la base de datos"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error inesperado: {str(e)}",
            "message": "No se pudo guardar la evaluación en la base de datos"
        }

# Función para recuperar evaluaciones
@tool
def get_patient_evaluation_by_id(paciente_id: int) -> Dict[str, Any]:
    """
    Recupera todas las evaluaciones asociadas a un paciente.

    Args:
        paciente_id: ID del paciente

    Returns:
        Dict con las evaluaciones o mensaje de error
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, paciente_id, nombre, telefono, correo,
                       historia_clinica, created_at, updated_at
                FROM evaluaciones
                WHERE paciente_id = ?
                ORDER BY created_at DESC
            ''', (paciente_id,))

            rows = cursor.fetchall()
            if not rows:
                return {
                    "success": False,
                    "message": f"No se encontraron evaluaciones para el paciente con ID {paciente_id}"
                }

            evaluaciones = []
            for row in rows:
                evaluaciones.append({
                    "id": row[0],
                    "paciente_id": row[1],
                    "nombre": row[2],
                    "telefono": row[3],
                    "correo": row[4],
                    "historia_clinica": row[5],
                    "created_at": row[6],
                    "updated_at": row[7]
                })

            return {
                "success": True,
                "evaluaciones": evaluaciones,
                "count": len(evaluaciones)
            }


    except sqlite3.Error as e:
        return {
            "success": False,
            "error": f"Error de base de datos: {str(e)}",
            "message": "No se pudo recuperar la(s) evaluación(es)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error inesperado: {str(e)}",
            "message": "Error al procesar la consulta"
        }


#----------------------------------
# AGENTE ENTREVISTADOR
#----------------------------------
from langgraph.prebuilt import create_react_agent

agent_interviewer = create_react_agent(
    model=llm,
    tools=[obtener_pregunta_por_id, psychological_diagnosis, save_patient_evaluation],
    prompt="""Eres el psicólogo clínico Freud, especializado en evaluaciones iniciales. Tu rol es administrar un cuestionario terapéutico con empatía y precisión clínica. Sigue estas instrucciones:

1. Presentate siempre, explica el proceso de una evaluación incicial y pide consentimiento para continuar.
2. Si el usuario no da su consentimiento, despidete y finaliza la sesión.
3. Si el usuario da su consentimiento, inicia la entrevista.
4. Utiliza estrcitamente las preguntas de la base de datos, no inventes las preguntas.
5. Accede a la base de datos y toma preguntas de la tabla preguntas accediendo a cada una de ellas por medio de un índice que vas a ir aumentando en cada interacción con el usuario(Utiliza SIEMPRE TODAS las preguntas de la base de datos en el orden dispuesto).
6. Humaniza cada pregunta y siempre sé muy empatico con el usuario(No enumeres las preguntas al paciente ej: pregunta 1: ¿cómo te sientes hoy?).
7. Si el paciente no cumple con el condicional de la pregunta, pasa a la sigiente pregunta.
8. Evalúa si la respuesta del usuario es válida y útil (es decir, relevante y no evasiva, como "No sé" o respuestas vacías).
9. Si la respuesta no es válida, reformulala y muestrala al usuario.
10. Cuando termines de realizar TODAS las preguntas de la base de datos, crea un caso clínico coherente en base a las respuestas del usuario.
11. Con el caso clínico genera un diagnostico con la herramienta disponible. Muestra el diagnostico al usuario de una manera humanizada.
12. En base al caso clínicio y al diagnóstico generado, crea una historia clínica profesional.
13. Guarda la información del usuario en la base de datos.
""",
name="agent_interviewer"
)

#---------------------------------
# AGENTE DE CONSULTA
#---------------------------------

from langgraph.prebuilt import create_react_agent

queries_agent = create_react_agent(
    model=llm,
    tools=[get_patient_evaluation_by_id],
    prompt="""Eres un asistente llamado Élia, especializado en consultas a la base de datos. asistir a médicos en la consulta de información de pacientes.

1. Presentate siempre.
2. Responde siempre de manera profesional, clara y estructurada.
3. Nunca simules la llamada a una herramienta, utiliza la herramienta que tienes disponible para la consulta.
4. Cuando termines de responder preguntas, finaliza la sesión.""",
name="queries_agent"
)

#----------------------------------
# FUNCIONES PARA REGISTRO
#----------------------------------

# Configuración de la base de datos
DB_PATH = "psychology_system.db"

def get_db_connection():
    """Obtiene una conexión a la base de datos"""
    return sqlite3.connect(DB_PATH)

def hash_password(password: str) -> str:
    """Hashea una contraseña usando SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_database():
    """Inicializa las tablas de la base de datos si no existen"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Tabla de PACIENTES
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Tabla de MÉDICOS
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medicos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Tabla de EVALUACIONES
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER NOT NULL,
            nombre TEXT NOT NULL,
            telefono TEXT,
            correo TEXT,
            historia_clinica TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def register_user(username: str, password: str, role: str) -> bool:
    """Registra un nuevo usuario en la base de datos"""
    conn = get_db_connection()
    cursor = conn.cursor()

    hashed_password = hash_password(password)
    table_name = "pacientes" if role == "paciente" else "medicos"

    try:
        cursor.execute(f'''
            INSERT INTO {table_name} (username, password)
            VALUES (?, ?)
        ''', (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def authenticate_user(username: str, password: str, role: str) -> bool:
    """Autentica un usuario"""
    conn = get_db_connection()
    cursor = conn.cursor()

    hashed_password = hash_password(password)
    table_name = "pacientes" if role == "paciente" else "medicos"

    cursor.execute(f'''
        SELECT id FROM {table_name}
        WHERE username = ? AND password = ?
    ''', (username, hashed_password))

    result = cursor.fetchone()
    conn.close()

    return result is not None

# Estado inicial completo requerido por agent_interviewer
initial_state: agentState = {
    "messages": []  # se inicializa vacío
}

#---------------------------------
#  INICIALIZACIÓN DE LA SESIÓN
#---------------------------------
def init_session_state():
    #####
    if 'screen' not in st.session_state:
        st.session_state.screen = "login"  # valores: 'login', 'register', 'chat'
    #####
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = initial_state.copy()

#----------------------------------
# AUTENTICACIÓN
#----------------------------------
def show_login_screen():
    st.title("🔐 Iniciar Sesión")
    with st.form("login_form"):
        login_username = st.text_input("Usuario", key="login_user")
        login_password = st.text_input("Contraseña", type="password", key="login_pass")
        login_role = st.selectbox("Rol", ["paciente", "médico"], key="login_role")
        login_btn = st.form_submit_button("Iniciar Sesión")

        if login_btn:
            if login_username and login_password:
                if authenticate_user(login_username, login_password, login_role):
                    st.session_state.authenticated = True
                    st.session_state.user_role = login_role
                    st.session_state.username = login_username
                    st.success("¡Bienvenido!")
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas.")
            else:
                st.error("Por favor, completa todos los campos.")
    
    if st.button("¿No tienes cuenta? Regístrate"):
        st.session_state.screen = "register"
        st.rerun()

def show_register_screen():
    st.title("📝 Registro de Usuario")
    with st.form("register_form"):
        reg_username = st.text_input("Usuario", key="reg_user")
        reg_password = st.text_input("Contraseña", type="password", key="reg_pass")
        reg_role = st.selectbox("Rol", ["paciente", "médico"], key="reg_role")
        register_btn = st.form_submit_button("Registrarse")

        if register_btn:
            if reg_username and reg_password:
                if register_user(reg_username, reg_password, reg_role):
                    st.success("¡Usuario registrado exitosamente!")
                    st.session_state.screen = "login"
                    st.rerun()
                else:
                    st.error("El usuario ya existe. Intenta con otro nombre.")
            else:
                st.error("Por favor, completa todos los campos.")
    
    if st.button("¿Ya tienes cuenta? Inicia sesión"):
        st.session_state.screen = "login"
        st.rerun()

#----------------------------------
# INTERFAZ DE CHAT
#----------------------------------
def show_chat_interface():
    """Muestra la interfaz de chat"""
    # Header con info del usuario
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        role_emoji = "🏥" if st.session_state.user_role == "médico" else "👤"
        st.markdown(f"#### {role_emoji} Chat - {st.session_state.user_role.title()}")
    with col2:
        st.markdown(f"**Usuario:** {st.session_state.username}")
    with col3:
        if st.button("Cerrar Sesión"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.session_state.chat_history = []
            st.session_state.agent_state = initial_state.copy()
            st.rerun()

    st.divider()

    # Mostrar historial de mensajes desde agent_state["messages"]
    for message in st.session_state.agent_state["messages"]:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        elif isinstance(message, AIMessage):
            if message.content and message.content.strip():
                st.chat_message("assistant").write(message.content)
        elif isinstance(message, ToolMessage):
            if message.content and message.content.strip():
            # Optionally display tool messages for debugging, but typically hidden
                st.chat_message("assistant").write(f"Tool output: {message.content}")
                pass

    # Entrada de texto del usuario
    user_input = st.chat_input("Escribe tu mensaje aquí...")

    if user_input:
        # Agregar mensaje del usuario al estado
        user_msg = HumanMessage(content=user_input)
        st.session_state.agent_state["messages"].append(user_msg)
        st.chat_message("user").write(user_msg.content)

        # Ejecutar el agente con el estado completo
        if st.session_state.user_role == "paciente":
            result = agent_interviewer.invoke(st.session_state.agent_state)
        else:
            result = queries_agent.invoke({"messages": st.session_state.agent_state["messages"]})

        # Actualizar el estado completo
        st.session_state.agent_state.update(result)

        # Obtener la lista de mensajes
        messages = result.get("messages", [])

        # Filtrar solo los que son del tipo AIMessage
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage) and msg.content and msg.content.strip()]


        # Verificar que haya al menos uno
        if ai_messages:
            last_msg = ai_messages[-1]  # Tomar el último

            # Guardar en historial
            #st.session_state.chat_history.append(last_msg)

            # Mostrarlo en pantalla
            st.chat_message("assistant").write(last_msg.content)
#
def main():
    """Función principal de la aplicación"""
    # Inicializar base de datos y estado
    init_database()
    init_session_state()

    # Mostrar interfaz según el estado de autenticación
    #if not st.session_state.authenticated:
    #    show_auth_screen()
    #else:
    #    show_chat_interface()

    if st.session_state.authenticated:
        show_chat_interface()
    else:
        if st.session_state.screen == "login":
            show_login_screen()
        elif st.session_state.screen == "register":
            show_register_screen()


if __name__ == "__main__":
    main()