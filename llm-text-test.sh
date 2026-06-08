#!/bin/bash


curl -s -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
        "messages": [
	    {"role": "system", "content": "keep answers short. No emojis, no em dashes. Be fair."},
	    {"role": "user", "content": "'"$1"'"}
        ],
        "max_tokens": 500
    }' | jq -r '.choices[0].message.content'


