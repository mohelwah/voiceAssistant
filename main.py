import langchain
import openai
import elevenlabs
from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from elevenlabs import set_api_key
from elevenlabs import generate, stream

from flask import Flask, render_template, request
from playsound import playsound
import tempfile
import os

set_api_key("")


load_dotenv(find_dotenv())


# LLM

llm = ChatOpenAI()

# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)


# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
def get_replay(question: str, conversation: LLMChain) -> str:
    result = conversation({"question": question})
    return result


def save_temp_mp3(data):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "temp_sound.mp3")

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(data)

    return temp_file_path


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form["message"]
        processed_message = get_replay(message, conversation)
        audio_result = generate(processed_message["text"])
        audio_file = save_temp_mp3(audio_result)

        # Play a sound (adjust the sound file path accordingly)
        playsound(audio_file)

        return render_template("index.html", processed_message=processed_message)

    return render_template("index.html", processed_message=None)


if __name__ == "__main__":
    app.run(debug=True)
