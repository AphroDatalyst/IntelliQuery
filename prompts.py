"""
Prompt templates for the IntelliQuery RAG system.
Includes system prompts, user prompts, and evaluation prompts.
"""

from langchain_core.prompts import ChatPromptTemplate

# SYSTEM PROMPTS
SYSTEM_PROMPT_BASE = """You are a helpful assistant that answers strictly using the provided context. 
                        Only use information explicitly stated in the context. 
                        If the answer is not present, reply exactly: Not found in the provided documents. 
                        Be concise and cite specific details when possible."""

SYSTEM_PROMPT_WITH_MEMORY = """You are a helpful assistant with access to conversation history and retrieved context.
                                - Answer ONLY the CURRENT question asked by the user. Do NOT repeat or summarize previous answers.
                                - Use the provided context and previous conversation history ONLY to understand references and context.
                                - If information is not in the context or history, reply exactly: Not found in the provided documents.
                                - When answering follow-up questions, use previous Q&A pairs to resolve references (like "it", "that", "they") but answer only the current question.
                                - Be concise, accurate, and cite specific details when possible."""

# USER PROMPT TEMPLATES
USER_PROMPT_SINGLE_TURN = """Question: {question}

                            Context:
                            {context}

                            Answer concisely:"""

USER_PROMPT_MULTI_TURN = """Previous conversation (for context only - do NOT repeat these answers):
                            {conversation_history}

                            Current question to answer: {question}

                            Context from documents:
                            {context}

                            Answer ONLY the current question above. Do not repeat previous answers."""

# CHAT PROMPT TEMPLATES
def get_single_turn_prompt() -> ChatPromptTemplate:
    """Get prompt template for single-turn Q&A (no memory)."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_BASE),
        ("human", USER_PROMPT_SINGLE_TURN),
    ])


def get_multi_turn_prompt() -> ChatPromptTemplate:
    """Get prompt template for multi-turn conversation (with memory)."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_WITH_MEMORY),
        ("human", USER_PROMPT_MULTI_TURN),
    ])

# EVALUATION PROMPTS (for RAGAS-style evaluation)
EVAL_PROMPT_ANSWER_RELEVANCE = """Given the question and answer, determine if the answer is relevant to the question.
                                Question: {question}
                                Answer: {answer}
                                Rate the relevance on a scale of 0-1, where 1 is highly relevant and 0 is not relevant."""

EVAL_PROMPT_GROUNDEDNESS = """Given the answer and context, determine if the answer is grounded in the context.
                                Answer: {answer}
                                Context: {context}
                                Rate the groundedness on a scale of 0-1, where 1 is fully grounded and 0 is not grounded."""

EVAL_PROMPT_CONTEXT_PRECISION = """Given the question and retrieved context, determine if the context is relevant.
                                Question: {question}
                                Context: {context}
                                Rate the precision on a scale of 0-1, where 1 is highly precise and 0 is not precise."""

EVAL_PROMPT_CONTEXT_RECALL = """Given the question, answer, and context, determine if all relevant information from context is used.
                                Question: {question}
                                Answer: {answer}
                                Context: {context}
                                Rate the recall on a scale of 0-1, where 1 is complete recall and 0 is no recall."""

