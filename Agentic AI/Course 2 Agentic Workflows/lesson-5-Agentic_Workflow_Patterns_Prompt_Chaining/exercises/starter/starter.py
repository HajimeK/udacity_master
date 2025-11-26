import os
from openai import OpenAI

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
    def __init__(self, llm: LLM):
        self.llm = llm
        self.behavior = ""

    def set_agent_behavior(self, behavior: str):
        self.behavior = behavior

    def query(self, user_prompt: str):
        llm.set_system_prompt(self.behavior)
        return self.llm.query(user_prompt)

llm = LLM("gpt-3.5-turbo")

feedstock_analyst_agent = Agent(llm)
feedstock_analyst_agent.set_agent_behavior(
    """You are a petrochemical expert analyzing hydrocarbon feedstocks.
            Provide a concise analysis of the given feedstock,
            highlighting its key components and general suitability
            for producing valuable refined products like gasoline,
            diesel, and kerosene.""")

distillation_planner_agent = Agent(llm)
distillation_planner_agent.set_agent_behavior(
    """You are a refinery distillation tower operations planner.
        Based on the provided feedstock analysis,
        estimate the potential percentage yields for major products like gasoline,
        diesel, and kerosene. Be realistic.""")

product_list_agent = Agent(llm)
product_list_agent.set_agent_behavior(
    "List all products from the distillation plan above.")

market_analyst_agent = Agent(llm)
market_analyst_agent.set_agent_behavior(
    """You are a market analyst for petroleum products.
        Provide current demand and pricing insights.
        Format the response as:
        # PRODUCT DEMAND ANALYSIS
        # PRICE TRENDS
        # MARKET RECOMMENDATIONS""")

production_optimizer_agent = Agent(llm)
production_optimizer_agent.set_agent_behavior(
    """You are a refinery production strategist.
        Make a recommendation balancing operational output and market demand.
        Include:
        # OPTIMIZED PRODUCTION PLAN
        # RATIONALE
    # STRATEGIC NOTES""")

feedstock_name = "Light Sweet Crude"
feedstock_analysis = feedstock_analyst_agent.query(f"Analyze the feedstock: {feedstock_name}")
distillation_plan = distillation_planner_agent.query(f"Based on this feedstock report, plan the distillation:\n\n{feedstock_analysis}")
product_list = product_list_agent.query(distillation_plan)
market_data = market_analyst_agent.query(f"Analyze market conditions for the following products:\n{product_list}")
production_plan = production_optimizer_agent.query(
    f"""Use the following inputs to recommend a production plan:

Distillation Plan:
{distillation_plan}

Market Analysis:
{market_data}
"""
)

print("end")

