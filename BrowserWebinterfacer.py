import requests
OPENAI_URL = 'https://api.openai.com/v1/chat/completions'
API_KEY = ''
def slownonstream_openai_api(messages):
    max_retries = 3
    retries = 0

    while retries < max_retries:
        openai_data = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.3,
            "top_p": 1,
            "presence_penalty": 0.8,
            "frequency_penalty": 0.9,
        }

        openai_headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'}
        openai_response = None
        try:
            openai_response = requests.post(OPENAI_URL, headers=openai_headers, json=openai_data, timeout=22 )
        except requests.exceptions.Timeout:
            retries += 1
            continue

        if openai_response and openai_response.ok:
            return openai_response.json()['choices'][0]['message']['content']

        print(f'Error: OpenAI request failed with status code {openai_response.status_code} and message: {openai_response.text}')
        return None

    print(f'Error: Reached maximum retries ({max_retries}) without success.')
    return None



def bingresult_summary(prompt, results):
    messages = [
        {"role": "system", "content": "You are a summarizer/selector for web search results."},
        {"role": "system", "content": "You are given the search query and 10+ results from a web search engine."},
        {"role": "system", "content": "You must choose the 3 best results that are most relevant to the query."},
        {"role": "user", "content": "return the chosen results in the same format as the results. include snippet and url."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": results},
    ]
    
    
    response = slownonstream_openai_api(messages)
    print (response)
    return response


def webpage_summary(pagetext):
    messages = [
        {"role": "system", "content": "You are a summarize for the text extracted from a web page"},
        {"role": "system", "content": "You are given the title and the text extracted from a web page"},
        {"role": "system", "content": "You must summarize the text preserving as much of the original information as possible"},
        {"role": "user", "content": "return the results as a couple of paragraphs."},
        {"role": "user", "content": pagetext},
    ]
    response = slownonstream_openai_api(messages)
    print (response)
    return response

