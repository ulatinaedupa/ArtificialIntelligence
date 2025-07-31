from langchain_openai import ChatOpenAI
from langchain_core.messages.system import SystemMessage
from langgraph.prebuilt import create_react_agent
from state_account_agent.tools import state_account_tools

model=ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = SystemMessage(
    content="Eres un asistente de atención al cliente. Responde a las preguntas de manera rápida y natural, " \
            "sin enumerar, solo respondiendo lo preguntado, detalla la moneda que es dólares y el día de ocurrencia."
)
state_account_agent = create_react_agent(model=model, tools=state_account_tools, prompt=prompt)