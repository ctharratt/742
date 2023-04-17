import openai
import os

openai.api_key =  "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"

model_engine = "text-davinci-002"
prompt = "Hello, how are you doing today?"

response = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=60,
    n=1,
    stop=None,
    temperature=0.5,
)

print(response.choices[0].text.strip())

print(response)
