# Query-Intelligence
Overview :
This project implements a query processing system capable of answering user questions about companies by scraping relevant information from URLs. The system integrates several AI models, including LLMs for natural language understanding and response generation, semantic search models such as SBERT, and various function to handle a variety of query types efficiently.

Key Features :
LLM Integration: The system uses advanced LLMs for generating and completing natural language responses. These models help understand and generate contextually relevant answers to user queries.
Multi-Model Approach: In addition to LLMs, the system incorporates SBERT for semantic similarity matching, and FAISS for fast indexing and retrieval of pre-processed information.
Sentence Completion: The system detects incomplete responses and uses an LLM to complete sentences, ensuring logical and grammatically correct answers.
Contextual Query Handling: The combination of LLMs and semantic models allows for dynamic query handling, enabling the system to understand and address user requests, even with complex sentence structures.

How to Use the Application :
Follow these steps to use the Query Intelligence system:
1.	Enter the URL(s):
  o	Input the website URL(s) you wish to query. Ensure that each URL includes the https:// protocol and belongs to the company's domain.
    	Example: https://website.com/
  o	You can enter multiple URLs if needed.

2.	Ask Your Question:
  o	In the provided input field, type the question you want to ask about the content of the website(s).
  o	There are no specific limitations on the type of questions, but clarity and specificity are encouraged for optimal results.

3.	View the Response:
  o	The system will process your query and display the response directly within the application interface.
  o	Note: Execution time depends on the website content and may vary. Please be patient as it may take some time to generate the results.

4.	Download Responses:
  o	After the responses are displayed in the UI, you can also download them in a spreadsheet format.
  o	The spreadsheet will include columns for the URL, Queries, and Answers for easy reference.

Models and Tools Used :
This project uses a combination of advanced language models, machine learning techniques, and natural language processing tools to handle user queries efficiently.
1. Large Language Models (LLMs)
The system utilizes Hugging Face LLMs like Mistral-Nemo-Instruct-2407 to understand and generate human-like responses. These models interpret user queries, complete sentences, and     resolve ambiguities in the responses.
2. SBERT (Sentence-BERT)
SBERT (Sentence-BERT) is used for semantic search. It matches user queries with pre-defined reference queries by converting them into embeddings and calculating similarity scores to find the best match. The model used is paraphrase-MiniLM-L6-v2.
3. Text Embeddings and Search
Hugging Face Embeddings help generate vector representations of text, which are crucial for comparing queries and documents. Combined with FAISS (Facebook AI Similarity Search), the system quickly retrieves relevant information from large datasets.
4. Named Entity Recognition (NER)
The system uses SpaCy’s en_core_web_sm model for recognizing key entities like company names, locations, and products from the text, helping to refine the answers given to users.
5. Punctuation and Sentence Correction
Deep Multilingual Punctuation Model automatically restores punctuation in user inputs and responses, ensuring outputs are clear and grammatically correct.
NLTK’s PunktSentenceTokenizer is used to break text into individual sentences, assisting in text analysis and query processing.
6. Web Scraping and Data Retrieval
Requests and BeautifulSoup are used for scraping relevant information from websites. This data is processed by the system to generate responses for user queries.
7. Text Processing
The system uses Langchain’s RecursiveCharacterTextSplitter to split large pieces of text into manageable chunks, which are easier to process, especially for long documents.

