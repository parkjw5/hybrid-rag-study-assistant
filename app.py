from core.generator import RAGGenerator

def main():
    print("Hybrid RAG System")
    print("Type 'exit' to quit\n")

    generator = RAGGenerator()

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        # ✅ Use answer() NOT generate()
        answer, results, confidence = generator.answer(question)

        print("\nAnswer:\n")
        print(answer)

        print("\nConfidence:", confidence)
        print("-" * 50)


if __name__ == "__main__":
    main()