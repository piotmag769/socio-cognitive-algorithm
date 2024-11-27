import random
from typing import Type, Sequence

from .agents.base import BaseAgent
from collections import defaultdict


class ExchangeMarket:
    def __init__(self, agents: Sequence[Type[BaseAgent]], migration: bool = False):
        self.agents = agents
        self.migration = migration

    def exchange_information(self):
        was_paired = defaultdict(bool)
        for agent in self.agents:
            if was_paired[agent]:
                continue

            was_paired[agent] = True
            not_paired_agents = [
                agent for agent in self.agents if not was_paired[agent]
            ]

            if len(not_paired_agents) == 0:
                break

            paired_agent = random.choice(not_paired_agents)
            was_paired[paired_agent] = True

            paired_agent_shared_solutions = paired_agent.get_solutions_to_share(agent)
            agent_shared_solutions = agent.get_solutions_to_share(paired_agent)

            if self.migration:
                paired_agent.remove_solutions(paired_agent_shared_solutions)
                agent.remove_solutions(agent_shared_solutions)

            agent.use_shared_solutions(
                paired_agent_shared_solutions, paired_agent
            )
            
            paired_agent.use_shared_solutions(
                agent_shared_solutions, agent
            )
