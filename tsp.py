from jmetal.problem.singleobjective.tsp import TSP

if __name__ == "__main__":
    # 629 is optimal
    tsp = TSP("tsp.txt")
    print(tsp.distance_matrix)
