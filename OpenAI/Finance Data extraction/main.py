import openai
openai.api_key = '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'

def chat_with_gpt():
    prompt = 'write a small paragraph about newyork'
    response= openai.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages= [
            {"role": "user", "content":'write a small paragraph about newyork'}

        ]
    )
    print(response.choices[0]['message']['content'])
if __name__ == '__main__':
    chat_with_gpt()