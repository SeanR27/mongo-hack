import voyageai
import numpy as np

key = ""
vo = voyageai.Client(api_key = key)

documents = [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
    "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
    "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
    "Apple’s conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
    "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
]

response = vo.embed(documents, model="voyage-4-large", input_type="document")
embedded_docs = response.embeddings

query = "When is Apple's conference call scheduled?"
query_embd = vo.embed([query], model="voyage-4-large", input_type="query").embeddings[0]

def get_k_nearest_neighbors(query_embd, doc_embds, documents, k=3):

    # 1. Compute cosine similarity for every vector in doc_embds against query_embd
    similarities = np.dot(doc_embds, query_embd)
    
    # np.argsort() returns an list of INDICES by element's ascending order value 
    sorted_indices_asc = np.argsort(similarities)
    
    # reverese the array to get descending order (highest similarity first)
    sorted_indices = sorted_indices_asc[::-1]
        
    # 3. Take the top k indices
    top_k_indices = sorted_indices[:k]
    
    # 4. Grab the corresponding documents and their scores
    results = []
    for idx in top_k_indices:
        results.append({
            "index": int(idx),
            "document": documents[idx],
            "score": float(similarities[idx])
        })
        
    return results


results = get_k_nearest_neighbors(query_embd, embedded_docs, documents, 6)

for r in results:
    print("Index: " + str(r["index"]))
    print("Document: " + str(r["document"]))
    print("Score: " + str(r["score"]))
    print()