from openai import OpenAI




def LLM_demo(human_input, model_name="gpt-4o-mini", max_tokens=4096, temperature=0.7, top_p=0.9):

    client = OpenAI(
    api_key="sk-5QyBNRgeFFiX6sY1aooYjvtygjNelFW87I6ziXkE6mP6tVeH", 
    base_url="http://35.220.164.252:3888/v1/")

    try:
        # 调用 Chat Completion API 并设置参数
        completion = client.chat.completions.create(
            model = model_name,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': human_input}],
            max_tokens=max_tokens,
            )
        return completion.model_dump()["choices"][0]["message"]["content"]
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    response1 = LLM_demo("你好，你是谁？")
    print(response1)
    response2 = LLM_demo("Who are you?")
    print(response2)