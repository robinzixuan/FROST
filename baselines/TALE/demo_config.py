demo_config = {
    "abbr": "GSM8K",
    "path": "./data/GSM8K-Zero",
    "name": "GSM8K-Zero",
    "reader_cfg": {
        "input_column": "question",
        "output_column": "answer"
    },
    "meta_prompt": {
        "round": [
            {
                "role": "HUMAN",
                "prompt": "Q: {question}\nPlease Give the response by strictly following this format: [[answer]], for example: Answer: [[50]].Let's think step by step:\n"
            },
            {
                "role": "BOT",
                "prompt": "A: {answer}"
            },
            {
                "role": "HUMAN",
                "prompt": "{reasoning_process_main}"
            },
            {
                "role": "HUMAN",
                "prompt": "{reasoning_process_socratic}"
            }
        ]
    }
}