import google.generativeai as genai

# Paste your actual key here
GEMINI_API_KEY = "AIzaSyDE1aLvDp7nDZmGlvfePzMLYMouEQ5t-Z4"
genai.configure(api_key=GEMINI_API_KEY)

print("Available models for generateContent:")
print("-" * 40)

# Loop through and print only the models that can generate text
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)