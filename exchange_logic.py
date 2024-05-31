import random
from typing import Type, Sequence

from agents import BaseAgent


class ExchangeMarket:
    def __init__(self, agents: Sequence[Type[BaseAgent]]):
        self.agents = agents

    def exchange_information(self):
        was_paired_map = {}
        for agent in self.agents:
            was_paired_map[agent] = True
            not_paired_agents = [
                ag for ag in self.agents if not was_paired_map.get(ag, False)
            ]

            if len(not_paired_agents) == 0:
                break

            paired_agent = random.choice(not_paired_agents)
            was_paired_map[paired_agent] = True

            agent.use_shared_solutions(
                paired_agent.get_solutions_to_share(agent), paired_agent
            )
            paired_agent.use_shared_solutions(
                agent.get_solutions_to_share(paired_agent), agent
            )
