from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")

model = ChatHuggingFace(llm=llm)


# create a state
class LLMState(TypedDict):
    question: str
    answer: str


def llm_qa(state: LLMState) -> LLMState:
    # extract the question from state
    question = state["question"]

    # formm a prompt
    prompt = f"answer the following question {question}"

    # ask the question to llm
    answer = model.invoke(prompt).content

    # update the answer in the state
    state["answer"] = answer

    return state


# define graph
graph = StateGraph(LLMState)

# add nodes
graph.add_node("llm_qa", llm_qa)

# add edges
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

# compile graph
workflow = graph.compile()


# execute graph
initial_state = {"question": "how far is sun from the earth?"}
final_state = workflow.invoke(initial_state)
print(final_state)
