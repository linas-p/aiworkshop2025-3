import ollama

response = ollama.chat(
    model='gemma3:12b',
    messages=[{
        'role': 'user',
        'content': 'Kas matoma paveikslėlyje?',
        'images': ['image.png']
    }]
)

print(response)