import os
from openai import OpenAI
#from dotenv import load_dotenv

# define OpenAI access helper class
class LLM:
    def __init__(self, model: str):
        path_to_key_file = os.path.join(os.path.abspath("."), "api_keys", "openai.key")
        key = open(path_to_key_file, "rt").read()
        vocareum_base_url = "https://openai.vocareum.com/v1"
        self.model = model

        self.client = OpenAI(
                        base_url = vocareum_base_url,
                        api_key=key
                    )
        print(self.client)

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def query(self, query: str):
        print(f"LLM System Prompt: {self.system_prompt}")
        print(f"LLM User Prompt: {query}")
        print(f"Calling LLM with model {self.model}")
        return self.client.chat.completions.create(
            model = self.model,
            messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query}
                    ],
            temperature=0
        ).choices[0].message.content

class Agent():
    def __init__(self,  llm: LLM, name = "", statement = ""):
        self.llm = llm
        self.behavior = ""
        self.name = name
        self.statement = statement

    def getName(self):
        return self.name

    def getStatement(self):
        return self.statement

    def set_agent_behavior(self, behavior: str):
        self.behavior = behavior

    def query(self, user_prompt: str):
        llm.set_system_prompt(self.behavior)
        return self.llm.query(user_prompt)

llm = LLM("gpt-3.5-turbo")

# --- Agents for Different Retail Tasks ---
product_researcher_agent = Agent(
    llm,
    "Product Researcher Agent",
    "Researches product specifications, market trends, and competitor pricing."
)
product_researcher_agent.set_agent_behavior(
    """You are a product research agent for a retail company. Your task is to provide
    structured information about products, market trends, and competitor pricing.""")

customer_analyzer_agent = Agent(
    llm,
    "Customer Analyzer Agent",
    "Analyzes customer feedback, preferences, and purchasing patterns."
)
customer_analyzer_agent.set_agent_behavior(
    """You are a customer analysis agent. Your task is to analyze customer feedback,
    preferences, and purchasing patterns."""
)

pricing_strategist_agent = Agent(
    llm,
    "Pricing Strategist Agent",
    "Recommends optimal pricing strategies based on research and analysis."
)
pricing_strategist_agent.set_agent_behavior(
    """You are a pricing strategist agent. Your task is to recommend optimal pricing
    strategies based on product research and customer analysis."""
)

# --- Routing Agent with LLM-Based Task Determination ---
class RoutingAgent(Agent):
    def __init__(self, llm: LLM):
        super().__init__(llm, context = [])
        self.agents = []

    def register_agent(self, agent: Agent):
        self.agents.append(agent)

    def set_agent_behavior(self):
        print([a.getName() for a in self.agents])
        registered_list =[f" - {a.getName()} : {a.getStatement()}\n" for a in self.agents]
        system_prompt = f"""You are an AI assistant that can route retail queries to the right agents.
You will be given a query, and your job is to determine the appropriate agent to handle it.
Agents available:\n
""" + "".join([f" - {a.getName()} : {a.getStatement()}\n" for a in self.agents]) + "Respond only with the agent's name, nothing else."

        super().set_agent_behavior(system_prompt)

    def agent_choice(self, query: str):
        agent_selected = self.query(f"Given the query: '{query}', which agent should handle this task?")
        # agent of the choice
        agents = list(filter(lambda A: A.getName() in agent_selected, self.agents))
        return agents[0]

router = RoutingAgent(llm)
router.register_agent(product_researcher_agent)
router.register_agent(customer_analyzer_agent)
router.register_agent(pricing_strategist_agent)
router.set_agent_behavior()


# --- Example Usage ---
if __name__ == "__main__":
    # Example queries
    queries = [
        "What are the specifications and current market trends for wireless earbuds?",
        "What do customers think about our premium coffee brand?",
        "What should be the optimal price for our new organic skincare line?"
    ]

    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("\nProcessing...")

        agent_chosen = router.agent_choice(query)
        print(type(agent_chosen))
        if "Product" in agent_chosen.getName():
            user_prompt = f"Research this product thoroughly: {query}"
        elif "Customer" in agent_chosen.getName():
            user_prompt = f"Analyze customer behavior for: {query}"
        elif "Pricing" in agent_chosen.getName():
            product_data = None
            if context and "product_data" in context:
                product_data = context["product_data"]
            else:
                print("Getting product information first...")
                product_data = product_researcher_agent(query)

            # Then, get customer insights
            customer_data = None
            if context and "customer_data" in context:
                customer_data = context["customer_data"]
            else:
                print("Getting customer insights...")
                customer_data = customer_analyzer_agent(query)

            user_prompt = f"""Recommend a pricing strategy for: {query}
            Product information: {product_data if product_data else 'No product data available'}
            Customer information: {customer_data if customer_data else 'No customer data available'}"""
        else:
            break
        result = agent_chosen.query(user_prompt)
        print("\nResult:")
        print(result)
        print("\n" + "-"*80)