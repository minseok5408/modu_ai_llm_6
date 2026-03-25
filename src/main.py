import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
question = "who are you?"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role" : "user", "content" : question},
    ]
)

answer = response.choices[0].message.content
print(answer)