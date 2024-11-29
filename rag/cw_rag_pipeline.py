import os
import openai
from minmaxsearch import Index

# Source code: https://machine-mind-ml.medium.com/discovering-semantic-search-and-rag-with-large-language-models-be7d9ba5bef4 
# reimplementation for my own practice of writing RAG pipelines
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = OPENAI_API_KEY)

# implement semantic search to retrieve relevant documents based on a query
def semantic_search(query):
    boost = {'question': 3, 'section': 0.5}
    retrieved = Index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )
    return retrieved

# builds a prompt that forms the context given relevant documents
def build_prompt(query, search_results):
    # build a template for the prompt
    prompt = """
    You're a course TA. Answer the QUESTION based on the CONTEXT.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    context = ""

    # iterate over the search results and build the context
    for doc in search_results:
        context += f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt.format(question=query, context=context)
    return prompt

# query the GPT-3 model with the prompt to generate an answer (generator function)
def query_gpt(prompt):
    output = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return output.choices[0].message.content

# run the RAG pipeline
def rag(query):
    search_results = semantic_search(query)
    prompt = build_prompt(query, search_results)
    answer = query_gpt(prompt)
    return answer

sample_query = "If I do not have a background in computer science, can I still take this course?"
print(rag(sample_query))