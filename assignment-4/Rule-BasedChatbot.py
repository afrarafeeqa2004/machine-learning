faqs = {
    "what are your timings": "We are open from 9 AM to 6 PM, Monday to Friday.",
    "what is your return policy": "You can return any product within 30 days of purchase.",
    "how can i contact support": "You can email us at support@example.com.",
    "do you offer refunds": "Yes, refunds are processed within 5-7 business days.",
    "where are you located": "We are located in Chennai, Tamil Nadu.",
}

def chatbot():
    print("FAQ Chatbot: Ask me something! Type 'bye' to exit.\n")

    while True:
        user = input("You: ").lower().strip()

        if user == "bye":
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Check if the user's question matches any FAQ
        found = False
        for question in faqs:
            if question in user:
                print("Chatbot:", faqs[question])
                found = True
                break

        if not found:
            print("Chatbot: Sorry, I don't know the answer to that yet.")

#run the chatbot
chatbot()
