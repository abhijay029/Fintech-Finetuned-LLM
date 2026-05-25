from transformers import pipeline

generator = pipeline("text-generation", model = "./model/fine_tuned_model")

def main():
    
    response = generator.generate("Explain EBITDA Response: ")

    print("Response of the Agent")
    print(response[0]["generated_text"])
    print(generator.device)

if __name__ == "__main__":
    main()
