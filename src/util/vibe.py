import anthropic
import os
import base64

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
MODEL = "claude-sonnet-4-5-20250929"


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_msg(text_queries: list[str], img_paths: list[str]):
    msg_content = []
    for query in text_queries:
        msg_content.append({"type": "text", "text": query})

    for img_path in img_paths:
        msg_content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_image(img_path),
                },
            }
        )
    messages = [{"role": "user", "content": msg_content}]
    return messages


def chat(messages: list[dict]):
    msg = client.messages.create(model=MODEL, max_tokens=13000, messages=messages)

    return msg.content
