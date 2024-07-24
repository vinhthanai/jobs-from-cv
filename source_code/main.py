from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_parse import LlamaParse
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.cohere import Cohere
PERSIST_DIR = "./storage"
MAX_ITER = 3
pdf_path = "input/data-science-cv-example.pdf"
query = ""
parser = LlamaParse(api_key = '')


import os 
os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(model="gpt-3.5-turbo")
agent = OpenAIAgent.from_tools(
    llm=llm,
    verbose=True,
)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)


def user_query(pdf_path: str, query: str):
    input_cv = parser.load_data(pdf_path)

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=10)
    response = query_engine.query(
        f"""
        You are a brilliant career adviser. Answer a question of job seekers with given information.\n
        If their CV information is given, use that information as well to answer the question.\n
        If you are not sure about the answer, return NA.\n
        You need to show the source nodes that you are using to answer the question at the end of your response.\n
        CV: {input_cv[0]}\n
        Question: {query}"""
    )
    return response


def evaluate_response(agent, query: str, response: str):
    prompt = f"""
    You are an evaluator of AI generated answers. Your task is to investigate whether the given response is logical and helpful to answer the query.\n
    First, given the query, context and response, give your binary judgement [Yes or No] if the response has good logic to answer the query in the context of the context.\n
    If the answer is 'Yes', return 'Yes' only.\n
    Second, if the answer of the first step is "No", re-write the query in the way it helps to retrieve more reverent context.\n
    In the case you are re-writing the query, return the re-written query only.\n
    Here is the query: {query}.\n
    Here is the response: {response}.\n
    """
    evaluation = agent.chat(prompt)
    return evaluation


def main(pdf_path, query, agent):
    i = 0
    response = user_query(pdf_path, query)
    evaluation = evaluate_response(agent, query, response.response)
    faithful_eval = faithfulness_evaluator.evaluate_response(response=response)
    relevancy_eval = relevancy_evaluator.evaluate_response(
        query=query, response=response
    )

    if ("Yes" in evaluation.response) | (faithful_eval.score >= 0.8) & (
        relevancy_eval.score >= 0.8
    ):
        return response.response
    else:
        while (i < MAX_ITER) & (faithful_eval.score < 0.8) & (
            relevancy_eval.score < 0.8
        ) | ("Yes" not in evaluation):
            query = evaluation.response  # Update query
            response = user_query(pdf_path, query)
            evaluation = evaluate_response(response.response)
            faithful_eval = faithfulness_evaluator.evaluate_response(response=response)
            i += 1
            if i == MAX_ITER:
                final_response = f"""Maximum iteration reached.\n
                Please be cautious with the generated response. It might not be helpful.\n
                {response.response}"""
                return final_response
        return response.response


if __name__ == "__main__":
    response = main(pdf_path, query, agent)
    print(response)
