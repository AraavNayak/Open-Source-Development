6. Vector Databases (AI databases)
Examples:
* Pinecone
* Weaviate
* FAISS (Facebook AI Similarity Search)
These are huge in AI.

Problem with normal databases
Suppose I search:
“movies about loneliness in space”
A SQL DB cannot understand meaning.
It only matches exact keywords.

Embeddings
LLMs convert text into vectors:

"cat" → [0.12, -0.88, ...]

Similar meanings end up near each other mathematically.
Vector DBs efficiently search:
“find vectors semantically similar to this one”
Used in:
* RAG
* AI search
* recommendations
* semantic retrieval


