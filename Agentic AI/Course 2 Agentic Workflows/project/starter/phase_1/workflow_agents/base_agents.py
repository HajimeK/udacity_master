# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI

#OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

class BaseAgent:
    def __init__(self, openai_api_key, model="gpt-3.5-turbo"):
        self.open_api_key = openai_api_key
        self.openai_base_url = OPENAI_BASE_URL
        self.client = OpenAI(
            base_url = self.openai_base_url,
            api_key=self.open_api_key
        )
        self.model = model
        self.system_prompt = ""
        self.user_prompt = ""
        self.messages = []

    def setModel(self, model: str):
        self.model = model

    def setAgentPersona(self, prompt: str):
        self.system_prompt = prompt
        self.messages.append({"role": "system", "content": self.system_prompt})

    def setUserQuery(self, prompt: str):
        self.messages.append({"role": "user", "content": self.system_prompt})

    def clearMessage(self):
        self.messages = []
        if 0 !=len(self.system_prompt):
            self.setAgentPersona(self.system_prompt)

    def respond(self):
        response = self.client.chat.completions.create(
            model = self.model,
            messages = self.messages,
            temperature=0
        )
        self.clearMessage()
        return response

# DirectPromptAgent class definition
class DirectPromptAgent(BaseAgent):

    def respond(self, prompt: str):
        self.setUserQuery(prompt)
        response = super().respond()
        return response.choices[0].message.content

# AugmentedPromptAgent class definition
class AugmentedPromptAgent(BaseAgent):
    def __init__(self, openai_api_key, persona: str):
        super().__init__(openai_api_key=openai_api_key)
        self.setAgentPersona(persona)

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        self.setUserQuery(input_text)
        response = super().respond()
        return  response.choices[0].message.content

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent(AugmentedPromptAgent):
    def __init__(self, openai_api_key, persona, knowledge):
        super().__init__(openai_api_key=openai_api_key, persona=persona)
        """Initialize the agent with provided attributes."""
        #           - The persona with the following instruction:
        #             "You are _persona_ knowledge-based assistant. Forget all previous context."
        #           - The provided knowledge with this instruction:
        #             "Use only the following knowledge to answer, do not use your own knowledge: _knowledge_"
        #           - Final instruction:
        #             "Answer the prompt based on this knowledge, not your own."
        self.knowledge = knowledge
        persona = f"""
        You are {persona} knowledge-based assistant. Forget all previous context.
        Use only the following knowledge to answer, do not use your own knowledge: {knowledge}
        Answer the prompt based on this knowledge, not your own.
        """
        self.setAgentPersona(persona)

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end >= len(text):
                break;

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions: int, model="gpt-3.5-turbo"):
        # Initialize the EvaluationAgent with given attributes.
        self.openai_api_key = openai_api_key
        self.openai_base_url = OPENAI_BASE_URL
        self.client = OpenAI(
            base_url = self.openai_base_url,
            api_key=self.openai_api_key
        )
        self.persone = persona
        self.evaluate_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.model = model

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        prompt_to_evaluate = initial_prompt

        for i in range(0,self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluate_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.persone},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.persone},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "number_of_iterations": i+1
        }

class RoutingAgent():

    def __init__(self, openai_api_key, agents = []):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.openai_base_url = OPENAI_BASE_URL
        self.embedding_model = "text-embedding-3-large"
        self.client = OpenAI(
            base_url = self.openai_base_url,
            api_key = self.openai_api_key
        )
        self.agents = agents
        self.last_ouput = ""

    def get_embedding(self, text):
        # Extract and return the embedding vector from the response
        response = self.client.embeddings.create(
            model = self.embedding_model,
            input = text,
            encoding_format = "float"
        )
        embedding = response.data[0].embedding
        return embedding

    # Define a method to route user prompts to the appropriate agent
    def route(self, user_input: str):
        # Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent['description'])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            # The best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_agent = agent
                best_score = similarity

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        self.last_ouput = best_agent["func"](user_input + "\n\nPrevious output:\n" + self.last_ouput)
        return self.last_ouput


class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.openai_base_url = OPENAI_BASE_URL
        # Instantiate the OpenAI client using the provided API key
        self.client = OpenAI(
            base_url = self.openai_base_url,
            api_key = self.openai_api_key
        )
        self.default_model = "gpt-3.5-turbo"
        self.knowledge = knowledge
        self.persona = f"""
        You are an action planning agent, and don't need to be like a human.
        Using your knowledge, you extract from the user prompt the steps requested
        to complete the action the user is asking for in the prompt.
        You return the steps as a list.
        Only return the steps in your knowledge which is defined as
        {self.knowledge}
        Forget any previous context.

        Output only the steps in the numbered list as the following format:
        1. Step 1
        2. Step 2
        3. Step 3
        4. step 4
        (continue until all steps are extracted)
        """


    def extract_steps_from_prompt(self, prompt):
        # Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"
        response = self.client.chat.completions.create(
                    model=self.default_model,
                    messages=[
                        {"role": "system", "content": self.persona},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
        response_text = response.choices[0].message.content

        # lean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")

        return steps
