from transformers import pipeline

generator = pipeline("text-generation", model = "./model/fine_tuned_model")

def main():
    
    response = generator("Explain EBITDA Response: ", max_length = 60)

    print("Response of the Agent")
    print(response[0]["generated_text"])


if __name__ == "__main__":
    main()
