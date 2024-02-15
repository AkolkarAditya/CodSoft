def simple_chatbot(user_input):
    # Convert user input to lowercase for case-insensitive matching
    user_input = user_input.lower()

    # Define predefined rules for responses
    if "hello" in user_input or "hi" in user_input:
        return "Hi there! How can I help you today?"

    elif any(greeting in user_input for greeting in ["good morning", "good afternoon", "good evening"]):
        return "Good to hear! What can I do for you?"

    elif "movie recommendation" in user_input:
        return "Sure, I can help with that. How about trying Inception?"

    elif "song recommendation" in user_input:
        return "Certainly! I recommend listening to 'Shape of You' by Ed Sheeran."

    elif "weather" in user_input:
        return "I'm not a weather bot, but did you know that the sun is actually a star?"

    elif "ok" in user_input or "okay" in user_input:
        return "Alright! Is there anything specific you would like to know or discuss?"

    else:
        return "Sorry, I'm not sure what you mean. Could you please rephrase?"

# Continuous conversation loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = simple_chatbot(user_input)
    print(f"Chatbot: {response}")
