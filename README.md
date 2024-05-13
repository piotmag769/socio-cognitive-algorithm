# Socio-cognitive algorithm

Algorithm description:
1. Every agent (`Agent`) manages the population of individuals that are subject to a genetic algorithm (`GeneticAlgorithm`).
2. There is some initial level of trust for him and his for other agents.
3. Agents exchange information about their populations (via `ExchangeMarket`). They can manipulate what part of their population are they revealing to other agents based on various things, such as the level of trust. There will be also a possibility of deception, that may bring an advantage over other agents in the race of receiving the best fitness faster. Various strategies of behavior shall be tested.
4. After receiving information about other population an agent can do one of few things with it, i.e. reject it, include it into his population, crossover with his population and use socio-cognitive learning operator to try to mimic this population. The comparison between effects brought by the use of different variation operators will be made. The choice of one operator over the other can also be a part of a characteristic of agents (something like an attitude).
5. The level of trust is adjusted after between-agent interaction.
