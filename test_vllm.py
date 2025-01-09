from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(model="demo_VITA_ckpt",  # "Qwen/Qwen2.5-1.5B-Instruct"
                                      prompt="San Francisco is a")
print("Completion result:", completion)
print("message:", completion.choices[0].text)

# chat_response = client.chat.completions.create(
#     model="demo_VITA_ckpt",  # "Qwen/Qwen2.5-1.5B-Instruct"
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me a joke."},
#     ]
# )
# print("Chat response:", chat_response)
# print("message:", chat_response.choices[0].message.content)
