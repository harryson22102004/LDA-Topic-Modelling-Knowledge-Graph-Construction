import re
from collections import defaultdict, Counter
 
STOPWORDS={'the','a','an','is','in','of','and','to','for','with','on','at','by','this','that','are','was'}
 
def preprocess(text):
    words=re.findall(r'\b[a-z]{3,}\b',text.lower())
    return [w for w in words if w not in STOPWORDS]
 
def lda_gibbs_simple(docs, n_topics=3, n_iter=50, alpha=0.1, beta=0.01):
    vocab=list(set(w for d in docs for w in d)); V=len(vocab)
    w2i={w:i for i,w in enumerate(vocab)}
    corpus=[[w2i[w] for w in d] for d in docs]
    n_dw=[[0]*n_topics for _ in range(len(docs))]
    n_wt=[[0]*n_topics for _ in range(V)]
    n_t=[0]*n_topics
    assignments=[]
    import random
    for d,doc in enumerate(corpus):
        z=[]
        for w in doc:
            t=random.randint(0,n_topics-1)
            z.append(t); n_dw[d][t]+=1; n_wt[w][t]+=1; n_t[t]+=1
           assignments.append(z)
    for _ in range(n_iter):
        for d,doc in enumerate(corpus):
            for i,w in enumerate(doc):
                t=assignments[d][i]
                n_dw[d][t]-=1; n_wt[w][t]-=1; n_t[t]-=1
                probs=[((n_dw[d][k]+alpha)*(n_wt[w][k]+beta)/(n_t[k]+V*beta)) for k in range(n_topics)]
                s=sum(probs); probs=[p/s for p in probs]
                t=random.choices(range(n_topics),probs)[0]
                assignments[d][i]=t; n_dw[d][t]+=1; n_wt[w][t]+=1; n_t[t]+=1
    topics=[]
    for k in range(n_topics):
        top=sorted(range(V),key=lambda w:-n_wt[w][k])[:5]
        topics.append([vocab[w] for w in top])
    return topics, assignments
 
docs=[preprocess(t) for t in [
    "Machine learning algorithms neural networks deep learning artificial intelligence",
    "Database query optimisation SQL indexing transactions relational model",
    "Graph algorithms shortest path spanning tree network flow complexity"]]
topics,_=lda_gibbs_simple(docs,3,20)
print("Discovered topics:")
for i,t in enumerate(topics): print(f"  Topic {i}: {t}")
