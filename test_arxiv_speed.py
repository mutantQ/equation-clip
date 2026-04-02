import arxiv
import time
import sys

print("Testing arXiv API speed...", flush=True)
start = time.time()

try:
    search = arxiv.Search(
        query="cat:quant-ph",
        max_results=5,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    print(f"Query created in {time.time()-start:.2f}s", flush=True)
    
    count = 0
    for result in search.results():
        count += 1
        print(f"Paper {count}: {result.title[:50]}... ({time.time()-start:.2f}s elapsed)", flush=True)
        
    print(f"\nTotal time for {count} papers: {time.time()-start:.2f}s", flush=True)
    print(f"Average: {(time.time()-start)/count:.2f}s per paper", flush=True)
    
except Exception as e:
    print(f"ERROR after {time.time()-start:.2f}s: {e}", flush=True)
    sys.exit(1)
