"""
Check for existing equation datasets we can use to accelerate research.
Options:
1. MathStackExchange dataset (questions with LaTeX equations)
2. arXMLiv dataset (arXiv papers already converted with equations extracted)
3. ProofWiki dataset 
4. Wikipedia math articles
"""

print("""
EXISTING EQUATION DATASETS TO CONSIDER:

1. arXMLiv (RECOMMENDED)
   - 1.3M+ arXiv papers already converted to XML/HTML
   - Equations pre-extracted with LaTeX source
   - URL: https://sigmathling.kwarc.info/resources/arxmliv-dataset-2020/
   - Size: ~200GB compressed
   
2. Math StackExchange Dump
   - ~2M questions with LaTeX equations
   - Community Q&A format (natural descriptions)
   - URL: https://archive.org/details/stackexchange
   
3. ProofWiki
   - Mathematical theorems with equations and descriptions
   - URL: https://proofwiki.org/wiki/Special:Export
   
4. OPTION: Use synthetic data for pilot
   - Generate equation-description pairs programmatically
   - Fast iteration for model validation
   - Then scale with real data

RECOMMENDATION: Start with option 4 (synthetic) to validate pipeline, 
then move to arXMLiv for real data.
""")
