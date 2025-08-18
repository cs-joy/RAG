import ollama


# Download dataset
dataset = []
with open('dataset/cat-facts.txt', 'r') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entities')

# Implement the vector database
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# The VECTOR_DB stores elements as (chunk, embedding) tuples, with embeddings represented by lists of floats (e.g., [0.1, 0.04, -0.34, 0.21, ...])
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')

# Implement the retrieval function
# Function for calculate cosine similarity between two vectors
def cosine_similarity(A, B):
    dot_product = sum([A_i * B_i for A_i, B_i in zip(A, B)])
    norm_A = sum([A_i ** 2 for A_i in A]) ** 0.5
    norm_B = sum([B_i ** 2 for B_i in B]) ** 0.5

    return dot_product / (norm_A * norm_B)

# Implement retrieval function - pre-trained retriever - p_eta
def retrieve_p_eta(query_q_x, top_K=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query_q_x)['embeddings'][0]
    # temporary list to store (chunk, similarity) paris
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    # sort by similarity in descending order, because higher similarity means more relevant
    similarities.sort(key=lambda x: x[1], reverse=True)

    # finally, return the top_K most relevant chunks
    return similarities[:top_K]

# Generation phrase - Generator - p_theta
input_query = str(input('Ask me a question: '))
retrieved_knowledge = retrieve_p_eta(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

chunk_text = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{chunk_text}
'''

# here we use `ollama`(pre-trained model) to generate the response. here we will use `instruction_prompt` as system message:
stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages = [
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
)

# print the response from the chatbot in real-time
print('Chatbot response: ')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)


