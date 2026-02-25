from transformers import pipeline

# Choose your model variant:
# For 0.1B (124M parameters, faster, more efficient):
model_path = "speakleash/Bielik-Guard-0.1B-v1.1"
# For 0.5B (443M parameters, higher accuracy):
# model_path = "speakleash/Bielik-Guard-0.5B-v1.1"

# Create pipeline for easy inference
classifier = pipeline("text-classification", model=model_path, token=True)

sample_texts = [
    "To jest normalna wiadomość bez problemów.",
    "Chcę się zabić i skończyć z tym wszystkim.",
    "Dlaczego hipisi są tacy brudni i głupi",
    "K*rwa, to jest ch*jowe rozwiązanie.",
    "Opowiem ci o seksie i intymności.",
    "Jak ukraść samochód i nie zostać złapanym.",
]

# Run predictions
for text in sample_texts:
    results = classifier(text, top_k=None)
    print(f"\nText: {text}")
    print("Safety scores:")
    for result in results:
        label = result["label"]
        score = result["score"]
        print(f"  {label}: {score:.3f}")

    max_score = max(result["score"] for result in results)
    print(f"  Overall safety risk: {max_score:.3f}")
