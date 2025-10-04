from openai.embeddings_utils import get_embedding, distances_from_embeddings
import pandas as pd
import tiktoken
import openai
import ast

openai.api_key = "" # YOUR API KEY

class EmbeddingChatbot():
    def __init__(self, 
    	embeddings_path, 
    	model_embeddings = "text-embedding-ada-002",
    	model_completion = "text-davinci-003",
    	max_prompt_tokens = 1200,
    	max_answer_tokens = 600):
        self.__df = pd.read_csv(embeddings_path)
        self.__df.embeddings = self.__df.embeddings.apply(ast.literal_eval)
        self.model_embeddings = model_embeddings
        self.model_completion = model_completion
        self.max_prompt_tokens = max_prompt_tokens
        self.max_answer_tokens = max_answer_tokens 

    def get_rows_sorted_by_relevance(self, question, df):
        """
        Function that takes in a question string and a dataframe containing
        rows of text and associated embeddings, and returns that dataframe
        sorted from least to most relevant for that question
        """
    
        # Get embeddings for the question text
        question_embeddings = get_embedding(question, engine=self.model_embeddings)
    
        # Make a copy of the dataframe and add a "distances" column containing
        # the cosine distances between each row's embeddings and the
        # embeddings of the question
        df_copy = df.copy()
        df_copy["distances"] = distances_from_embeddings(
        	question_embeddings,
        	df_copy["embeddings"].values,
        	distance_metric="cosine"
        	)
    
        # Sort the copied dataframe by the distances and return it
        # (shorter distance = more relevant so we sort in ascending order)
        df_copy.sort_values("distances", ascending=True, inplace=True)
        return df_copy


    def create_prompt(self, question, df, max_token_count):
        """
        Given a question and a dataframe containing rows of text and their
        embeddings, return a text prompt to send to a Completion model
        """
        # Create a tokenizer that is designed to align with our embeddings
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
        # Count the number of tokens in the prompt template and question
        prompt_template = """
        Answer the question based on the context below, and if the question
        can't be answered based on the context, say "I don't know"
        
        Context:

            {}
        
        ---

        Question: {}
        Answer:"""
    
        current_token_count = len(tokenizer.encode(prompt_template)) + \
                                len(tokenizer.encode(question))
    
        context = []
        for text in self.get_rows_sorted_by_relevance(question, df)["text"].values:
        
            # Increase the counter based on the number of tokens in this row
            text_token_count = len(tokenizer.encode(text))
            current_token_count += text_token_count
        
            # Add the row of text to the list if we haven't exceeded the max
            if current_token_count <= max_token_count:
                context.append(text)
            else:
                break

        return prompt_template.format("\n\n###\n\n".join(context), question)


    def answer_question(self, question, df, max_prompt_tokens, max_answer_tokens):
        """
        Given a question, a dataframe containing rows of text, and a maximum
        number of desired tokens in the prompt and response, return the
        answer to the question according to an OpenAI Completion model
    
        If the model produces an error, return an empty string
        """

        prompt = self.create_prompt(question, df, max_prompt_tokens)
        try:
            response = openai.Completion.create(
                model=self.model_completion,
                prompt=prompt,
                max_tokens=self.max_answer_tokens
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(e)
            return ""

    def predict(self, question, history):
    	return self.answer_question(question, self.__df, self.max_prompt_tokens, self.max_answer_tokens)