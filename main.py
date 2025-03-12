from llm import OpenAIChatbot
import gradio as gr

if __name__ == '__main__':
    chatbot = OpenAIChatbot(vector_store="all_messages")

    gr.ChatInterface(
        fn=chatbot.chat,
        type="messages"
    ).launch()
