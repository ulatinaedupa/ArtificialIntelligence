import gradio as gr
from helpers.chatter import EmbeddingChatbot

bot = EmbeddingChatbot(
    embeddings_path='data/embeddings.csv',
    model_embeddings = 'text-embedding-ada-002',
    model_completion ='text-davinci-003',
    max_prompt_tokens = 1800,
    max_answer_tokens = 600
    )

gr.ChatInterface(bot.predict).launch()