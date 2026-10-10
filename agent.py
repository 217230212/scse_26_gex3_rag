### COMPLETE THE CODE ###
"""University IT support agent.

Uses the qwen3-0.6b model together with a RAG tool backed by the
university IT policy knowledge base. The agent must answer questions
using only retrieved policy text and must NOT invent information
(such as a phone number that does not exist in the knowledge base).
"""

from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from rag_tool import rag_search


MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

tools = [rag_search]
SYSTEM_INSTRUCTION = (
    "You are a university IT support assistant. "
    "When the user asks about university IT matters, use the rag_search tool "
    "to look up the relevant policy text, then answer using ONLY the "
    "information returned by the tool. "
    "If the retrieved context does not contain the specific detail requested "
    "(for example a phone number), you MUST reply that you do not know. "
    "Never invent information, and never provide any response unrelated to the question."
)


try:
    react_prompt = hub.pull("hwchase17/react")
    prompt = react_prompt.partial(
        tools="\n".join([f"{t.name}: {t.description}" for t in tools]),
        tool_names=", ".join([t.name for t in tools]),
    )
except Exception:
  
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_INSTRUCTION + "\n\nTools:\n{tools}\nTool names: {tool_names}"),
            MessagesPlaceholder("agent_scratchpad", optional=True),
            ("human", "{input}"),
        ]
    )
    prompt = prompt.partial(
        tools="\n".join([f"{t.name}: {t.description}" for t in tools]),
        tool_names=", ".join([t.name for t in tools]),
    )

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def ask(question: str) -> str:
    """Ask the agent a question and return its final answer."""
    result = agent_executor.invoke({"input": question})
    return result.get("output", "")


if __name__ == "__main__":
    question = "What exact phone number should I call for the University IT Service Desk?"
    print("Q:", question)
    print("A:", ask(question))
