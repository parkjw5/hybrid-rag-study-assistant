import ollama

from config import LLM_MODEL, TOP_K
from core.retriever import HybridRetriever


class RAGGenerator:
    def __init__(self):
        self.retriever = HybridRetriever()

    # -----------------------------------
    # Generate Answer
    # -----------------------------------
    def answer(self, question: str, alpha: float = 0.5, top_k: int = TOP_K):

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=question,
            alpha=alpha,
            top_k=top_k
        )

        # If nothing retrieved → immediate abstention
        if not results:
            return (
                "I cannot find the answer in the provided materials.",
                [],
                0.0
            )

        # Build context with citations
        context_blocks = []

        for i, result in enumerate(results, start=1):
            citation_info = f"{result['metadata']['title']} (Page {result['metadata']['page']})"
            block = f"[{i}] Source: {citation_info}\n{result['text']}"
            context_blocks.append(block)

        context = "\n\n".join(context_blocks)

        # Strict grounding prompt
        prompt = f"""
You are a course assistant.

You must answer ONLY using the provided context.
Do NOT use outside knowledge.

If the answer is not explicitly supported by the context, respond exactly with:
"I cannot find the answer in the provided materials."

When you use information from a source, cite it using square brackets like:
[1], [2], etc.

You may cite multiple sources if needed.

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        answer_text = response["message"]["content"].strip()

        # Hard enforce exact refusal string
        if "I cannot find" in answer_text:
            answer_text = "I cannot find the answer in the provided materials."
        else:
            citation_list = []
            for i, result in enumerate(results, start=1):
                citation_info = f"[{i}] {result['metadata']['title']} (Page {result['metadata']['page']})"
                citation_list.append(citation_info)

            citations_formatted = "\n".join(citation_list)

            answer_text = f"{answer_text}\n\nSources:\n{citations_formatted}"

        confidence = results[0]["score"]

        return answer_text, results, confidence