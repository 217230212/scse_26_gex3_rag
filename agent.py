### COMPLETE THE CODE ###

from langchain_huggingface import HuggingFacePipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    do_sample=False,
    return_full_text=False,
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

tools = [rag_search]

system_prompt = (
    "You are a university IT support assistant. "
    "When the user asks about university IT matters, use the rag_search tool "
    "to look up the relevant policy text, then answer using ONLY the "
    "information returned by the tool. "
    "If the retrieved context does not contain the specific detail requested "
    "(for example a phone number), you MUST reply that you do not know. "
    "Never invent information, and never provide any response unrelated to the question."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt + "\n\nYou have access to the following tools:\n{tools}\n\nTool names: {tool_names}"),
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
    result = agent_executor.invoke({"input": question})
    return result.get("output", "")


if __name__ == "__main__":
    question = "What exact phone number should I call for the University IT Service Desk?"
    print("Q:", question)
    print("A:", ask(question))
