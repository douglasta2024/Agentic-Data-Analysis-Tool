import ollama

response = ollama.chat(model='gpt-oss:20b', messages=[
    {
        'role': 'user', 
          
        'content': 'Write a Python script to generate data visualization charts for csv data.'
     }
])

print(response['message']['content'])