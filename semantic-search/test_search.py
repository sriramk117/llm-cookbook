import minmaxsearch
import requests
import json

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

# create a list of documents
documents = []

for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# create an index object
index = minmaxsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

query = "Can I join the course if I have no prior experience?"

filter_dict = {"course": "data-engineering-zoomcamp"}
boost_dict = {"question": 3}

results = index.search(query, filter_dict, boost_dict, num_results=5)

for result in results:
    print(json.dumps(result, indent=2))