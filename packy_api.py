from openai import OpenAI

client = OpenAI(api_key="***REMOVED***", base_url="https://api-slb.packyapi.com/v1")

result = client.responses.create(
    model="gpt-5.1",
    input=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ],
    reasoning={ "effort": "high" }
)

print(result.output_text)