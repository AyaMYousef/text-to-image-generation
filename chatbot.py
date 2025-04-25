from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.agent import Agent
from agno.models.ollama import Ollama




memory_db = SqliteMemoryDb(table_name="memory", db_file="tmp/memory.db")
memory = Memory(db=memory_db)


agent = Agent(
    model=Ollama(id="llama3.2:latest"),
    instructions=["you will be given a prompt and you have to provide information and explanation about the prompt in 3 lines",
                "If the prompt contains a disease name, you have to provide information about the disease",
                "your task is to be like a guide for medical enthusiasts",],
    # num_history_responses=3,
    memory=memory,
    # enable_agentic_memory=True,
    add_history_to_messages=True,
    num_history_runs=3,
)

def get_agent_response(prompt: str, user_id: str = "john_doe@example.com" ):

    return agent.run(prompt , user_id=user_id).content