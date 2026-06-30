# from langgraph.graph import StateGraph, START, END
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.output_parsers import JsonOutputParser
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# from typing import TypedDict, Annotated
# import operator

# load_dotenv()


# class EvaluationSchema(BaseModel):
#     score: float = Field(description="Score out of 10")
#     feedback: str = Field(description="Feedback on essay")


# parser = JsonOutputParser(pydantic_object=EvaluationSchema)

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)


# essay = """ # Essay on Artificial Intelligence

# Artificial Intelligence, commonly known as AI, is one of the most important technologies of the modern world. It refers to the ability of machines and computers to perform tasks that usually require human intelligence, such as learning, problem-solving, decision-making, and understanding language. AI is transforming the way we live, work, and communicate.

# AI is used in many areas of our daily life. For example, voice assistants like Siri and Alexa use AI to understand and respond to human speech. Online platforms use AI to recommend movies, songs, and products based on user preferences. In healthcare, AI helps doctors diagnose diseases more accurately and quickly. In transportation, self-driving cars use AI to navigate roads and avoid obstacles.

# One of the biggest advantages of AI is efficiency. AI systems can process large amounts of data in a short time and perform repetitive tasks without getting tired. This helps improve productivity and reduces human effort. AI can also assist in dangerous environments, such as space exploration, disaster management, and industrial operations.

# However, AI also has challenges. Some people worry that AI may replace human jobs in certain industries. There are also concerns about privacy, security, and the ethical use of AI. Therefore, it is important to develop and use AI responsibly.

# In conclusion, Artificial Intelligence is a powerful technology that has the potential to improve human life in many ways. While it offers many benefits, it should be used carefully and ethically. As technology continues to grow, AI will play an even bigger role in shaping our future.
# """

# format_instructions = parser.get_format_instructions()

# prompt = f"""
# Evaluate the following essay.

# Return score and feedback.

# {format_instructions}

# Essay:
# {essay}
# """

# response = model.invoke(prompt)

# # print("Raw Output:")
# # print(response.content)

# parsed = parser.parse(response.content)

# print("\nParsed Output:")
# print(parsed)


# class UPSCState(TypedDict):

#     essay: str
#     language_feedback: str
#     analysis_feedback: str
#     clarity_feedback: str
#     overall_feedback: str
#     individual_scores: Annotated[list[int], operator.add]
#     avg_score: float


# def evaluate_language(state: UPSCState) -> UPSCState:
#     f"Evaluate the language quality of. the following essay an provide a feedback and assign a score out of 10 \n {state["essay"]}"
#     output = parsed.invoke(prompt)

#     return {"language_feedback": output.feedback, "individual_scores": [output.score]}


# def evaluate_analysis(state: UPSCState) -> UPSCState:
#     f"Evaluate the depth of analysis of the following essay an provide a feedback and assign a score out of 10 \n {state["essay"]}"
#     # output = parsed.invoke(prompt)
#     response = model.invoke(prompt)
# output = parser.parse(response.content)

#     return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}


# def evaluate_thought(state: UPSCState) -> UPSCState:
#     f"Evaluate the clarity of thought of the following essay an provide a feedback and assign a score out of 10 \n {state["essay"]}"
#     output = parsed.invoke(prompt)

#     return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}


# def final_evaluation(state: UPSCState) -> UPSCState:
#     prompt = f"Based on the following feedbacks create a summarized feedback \n lanuage feedback {state['language_feedback']} \n depth of analysis feedback - {state['analysis_feedback']} \n clarity of feedback {state['clarity_feedback']}"
#     overall_feedback = model.invoke(prompt).content

#     avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

#     return {"overall_feedback": overall_feedback, "avg_score": avg_score}


# # define graph
# graph = StateGraph(UPSCState)

# # add nodes
# graph.add_node("evaluate_language", evaluate_language)
# graph.add_node("evaluate_analysis", evaluate_analysis)
# graph.add_node("evaluate_thought", evaluate_thought)
# graph.add_node("final_evaluation", final_evaluation)

# # add edges
# graph.add_edge(START, "evaluate_language")
# graph.add_edge(START, "evaluate_analysis")
# graph.add_edge(START, "evaluate_thought")

# graph.add_edge("evaluate_language", "final_evaluation")
# graph.add_edge("evaluate_analysis", "final_evaluation")
# graph.add_edge("evaluate_thought", "final_evaluation")

# graph.add_edge("final_evaluation", END)

# workflow = graph.compile()

# initial_state = {"essay": essay}

# workflow.invoke(initial_state)


# 2nd solution

from dotenv import load_dotenv
from typing import TypedDict, Annotated, NotRequired
import operator
import re
import json

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser

# =========================================================
# 1. LOAD ENVIRONMENT
# =========================================================
load_dotenv()


# =========================================================
# 2. MODEL SETUP
# =========================================================
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# =========================================================
# 3. OUTPUT SCHEMA
# =========================================================
class EvaluationSchema(BaseModel):
    score: float = Field(description="Score out of 10")
    feedback: str = Field(description="Evaluation feedback")


parser = JsonOutputParser(pydantic_object=EvaluationSchema)
FORMAT_INSTRUCTIONS = parser.get_format_instructions()


# =========================================================
# 4. GRAPH STATE
# =========================================================
class UPSCState(TypedDict):
    essay: str

    language_feedback: NotRequired[str]
    analysis_feedback: NotRequired[str]
    clarity_feedback: NotRequired[str]

    overall_feedback: NotRequired[str]

    individual_scores: Annotated[list[float], operator.add]
    avg_score: NotRequired[float]


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================
def extract_json(text: str):
    """
    Extract JSON from messy LLM output.
    Handles cases like:
    Sure! Here's JSON:
    { ... }
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    return json.loads(match.group())


def run_evaluation(criteria: str, essay: str):
    """
    Generic evaluation function used by all evaluator agents.
    """

    prompt = f"""
You are an essay evaluator.

Evaluate the essay based on:
{criteria}

Return ONLY JSON.

{FORMAT_INSTRUCTIONS}

Essay:
{essay}
"""

    response = model.invoke(prompt)
    raw_output = response.content

    try:
        parsed = parser.parse(raw_output)
        return parsed

    except Exception:
        # Fallback if model adds extra text around JSON
        parsed = extract_json(raw_output)
        return parsed


# =========================================================
# 6. AGENT NODES
# =========================================================
def evaluate_language(state: UPSCState):
    output = run_evaluation(
        criteria="Language quality, grammar, spelling, vocabulary", essay=state["essay"]
    )

    return {
        "language_feedback": output["feedback"],
        "individual_scores": [output["score"]],
    }


def evaluate_analysis(state: UPSCState):
    output = run_evaluation(
        criteria="Depth of analysis, reasoning, critical thinking", essay=state["essay"]
    )

    return {
        "analysis_feedback": output["feedback"],
        "individual_scores": [output["score"]],
    }


def evaluate_clarity(state: UPSCState):
    output = run_evaluation(
        criteria="Clarity of thought, coherence, logical flow", essay=state["essay"]
    )

    return {
        "clarity_feedback": output["feedback"],
        "individual_scores": [output["score"]],
    }


def final_evaluation(state: UPSCState):
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    prompt = f"""
Summarize the following feedback into one final evaluation.

Language Feedback:
{state["language_feedback"]}

Analysis Feedback:
{state["analysis_feedback"]}

Clarity Feedback:
{state["clarity_feedback"]}
"""

    overall_feedback = model.invoke(prompt).content

    return {"overall_feedback": overall_feedback, "avg_score": avg_score}


# =========================================================
# 7. BUILD GRAPH
# =========================================================
graph = StateGraph(UPSCState)

graph.add_node("language", evaluate_language)
graph.add_node("analysis", evaluate_analysis)
graph.add_node("clarity", evaluate_clarity)
graph.add_node("final", final_evaluation)

graph.add_edge(START, "language")
graph.add_edge(START, "analysis")
graph.add_edge(START, "clarity")

graph.add_edge("language", "final")
graph.add_edge("analysis", "final")
graph.add_edge("clarity", "final")

graph.add_edge("final", END)

workflow = graph.compile()


# =========================================================
# 8. RUN
# =========================================================
# essay = """
# Artificial Intelligence (AI) is transforming modern society.
# It improves productivity and helps solve complex problems.
# However, ethical concerns such as bias and privacy remain.
# """

essay = """India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India"."""

# essay = """ # Essay on Artificial Intelligence

# Artificial Intelligence, commonly known as AI, is one of the most important technologies of the modern world. It refers to the ability of machines and computers to perform tasks that usually require human intelligence, such as learning, problem-solving, decision-making, and understanding language. AI is transforming the way we live, work, and communicate.

# AI is used in many areas of our daily life. For example, voice assistants like Siri and Alexa use AI to understand and respond to human speech. Online platforms use AI to recommend movies, songs, and products based on user preferences. In healthcare, AI helps doctors diagnose diseases more accurately and quickly. In transportation, self-driving cars use AI to navigate roads and avoid obstacles.

# One of the biggest advantages of AI is efficiency. AI systems can process large amounts of data in a short time and perform repetitive tasks without getting tired. This helps improve productivity and reduces human effort. AI can also assist in dangerous environments, such as space exploration, disaster management, and industrial operations.

# However, AI also has challenges. Some people worry that AI may replace human jobs in certain industries. There are also concerns about privacy, security, and the ethical use of AI. Therefore, it is important to develop and use AI responsibly.

# In conclusion, Artificial Intelligence is a powerful technology that has the potential to improve human life in many ways. While it offers many benefits, it should be used carefully and ethically. As technology continues to grow, AI will play an even bigger role in shaping our future.
# """

initial_state = {"essay": essay, "individual_scores": []}

result = workflow.invoke(initial_state)


# =========================================================
# 9. OUTPUT
# =========================================================
print("\nFINAL RESULT")
print("=" * 50)

print("Language Feedback:")
print(result["language_feedback"])

print("\nAnalysis Feedback:")
print(result["analysis_feedback"])

print("\nClarity Feedback:")
print(result["clarity_feedback"])

print("\nOverall Feedback:")
print(result["overall_feedback"])

print("\nIndividual_score:")
print(result["individual_scores"])

print(f"\nAverage Score: {result['avg_score']:.2f}/10")
