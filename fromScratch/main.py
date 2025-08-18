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
    VECTOR_DB.append(chunk, embedding)

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

##
