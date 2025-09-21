import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handles query decomposition and multi-step reasoning"""

    def __init__(self):
        self.reasoning_patterns = {
            "comparison": ["compare", "versus", "vs", "difference", "similar", "different"],
            "analysis": ["analyze", "examine", "evaluate", "assess", "study"],
            "explanation": ["explain", "how", "why", "what", "describe"],
            "synthesis": ["summarize", "combine", "integrate", "overview", "synthesis"],
            "factual": ["when", "where", "who", "which", "list"]
        }

    def classify_query_type(self, query: str) -> str:
        """Classify the type of query for better processing"""
        query_lower = query.lower()

        for query_type, keywords in self.reasoning_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type

        return "general"

    def decompose_complex_query(self, query: str) -> List[str]:
        """Break down complex queries into simpler sub-queries"""
        decomposed = []

        # Split on common conjunctions
        parts = re.split(r'\b(and|or|also|additionally|furthermore)\b', query, flags=re.IGNORECASE)

        for part in parts:
            part = part.strip()
            if part and part.lower() not in ['and', 'or', 'also', 'additionally', 'furthermore']:
                if part.count('?') > 1:
                    sub_parts = part.split('?')
                    for sub_part in sub_parts:
                        if sub_part.strip():
                            decomposed.append(sub_part.strip() + '?')
                else:
                    decomposed.append(part)

        if not decomposed:
            decomposed = [query]

        return decomposed

    def generate_reasoning_steps(self, query: str, retrieved_docs: List[Tuple]) -> List[str]:
        """Generate reasoning steps for the query"""
        steps = []
        query_type = self.classify_query_type(query)

        steps.append(f"1. Query Classification: Identified as '{query_type}' type query")
        steps.append(f"2. Document Retrieval: Retrieved {len(retrieved_docs)} relevant documents")

        if retrieved_docs:
            steps.append("3. Source Analysis:")
            for i, (doc, score) in enumerate(retrieved_docs, 1):
                steps.append(f"   - Document {i}: '{doc.title}' (similarity: {score:.3f}) from {doc.source}")

        steps.append("4. Information Synthesis: Combining relevant information from retrieved sources")
        steps.append("5. Answer Generation: Formulating comprehensive response based on available evidence")

        return steps