#!/usr/bin/env python3
"""
CLI entry point for RAG Agent.
"""

import argparse
import os
from .rag_agent import RAGAgent


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="RAG Agent with Ollama & LangChain")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ingest", "query", "interactive"],
        default="interactive",
        help="Operation mode"
    )
    parser.add_argument("--question", type=str, help="Question to ask")
    parser.add_argument("--model", type=str, default="llama2", help="Ollama model")
    parser.add_argument("--documents", type=str, default="docs", help="Documents folder")
    
    args = parser.parse_args()
    
    print(f"🚀 Initializing RAG Agent (model: {args.model})")
    agent = RAGAgent(model_name=args.model)
    
    if args.mode == "ingest":
        if not os.path.exists(args.documents):
            print(f"❌ Folder not found: {args.documents}")
            os.makedirs(args.documents, exist_ok=True)
            print(f"   Created. Add documents and run again.")
            return
        
        agent.load_documents(args.documents)
        print("✅ Documents ingested!")
        
    elif args.mode == "query":
        if not args.question:
            print("❌ Provide --question")
            return
        
        print(f"🔍 Query: {args.question}\n")
        result = agent.query(args.question)
        print(f"💬 {result['answer']}")
        if result.get('sources'):
            print(f"\n📚 Sources: {', '.join(result['sources'])}\n")
        
    elif args.mode == "interactive":
        print("\n" + "=" * 60)
        print("🤖 RAG Agent - Interactive Mode")
        print("=" * 60)
        print("Commands: 'exit' or 'quit' to end, 'clear' to clear history\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ["exit", "quit"]:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == "clear":
                    if agent.memory:
                        agent.clear_history()
                    print("🧹 History cleared.\n")
                    continue
                
                if not question:
                    continue
                
                print("🤔 Thinking...")
                result = agent.query(question)
                print(f"\n🤖 {result['answer']}")
                if result.get('sources'):
                    print(f"\n📚 Sources: {', '.join(result['sources'])}\n")
                else:
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

