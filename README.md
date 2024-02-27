# FlanT5-LLM Query Engine

<img width="955" alt="image" src="https://github.com/hamzahasan13/FlanT5-LLM-Query-Engine/assets/114373000/b95eb59a-6999-4d02-89ec-a68beb213d3c">

<img width="1451" alt="Working" src="https://github.com/hamzahasan13/FlanT5-LLM-Query-Engine/assets/114373000/551899bb-9bfe-4212-b98e-a0346c37d1fc">

Key Components:

**LangChain**: A framework for context-aware applications powered by language models, enabling intelligent reasoning and informed decision-making.

**Extracting-Content**:
•	Pdf object is created and is looped over to extract content from each page.
•	The content from the pages is divided into smaller pages of 1000 characters each. 
•	This is done to provide contextual information to LLM when a question is asked.
•	The embeddings work best with shorter pieces of text.
•	Instead of making the LLM read the entire book every time a question is asked, it is more efficient and cost-effective to give it a smaller section of relevant information to process.

**Embeddings**: are multi-dimensional numerical representations of meaning generated by language models. These embeddings capture semantic relationships between words and phrases.
•	Hugging Face maintains a repository of pre-trained transformer-based models.
•	Instructor-xl is an instruction finetuned text embedding model that is used to generate text embeddings.

**Vectorstore**: stores all the embeddings and performs vector search.
•	Facebook AI Similarity Search (FAISS) is built around an index type that stores a set of vectors and enables fast similarity search using nearest neighbor.

**Semantic-Search**:
•	Given a user query’s semantic representation, this algorithm identifies the most relevant documents based on their proximity in the embedding space.

**LLM**: Google’s Flan T5 (Text-to-Text Transfer Transformer) is an open-source, sequence-to-sequence, large language model. This LLM uses the output of the FAISS to generate an answer to the question asked by the user.

References:
- https://huggingface.co/hkunlp/instructor-large
- https://bennycheung.github.io/ask-a-book-questions-with-langchain-openai
