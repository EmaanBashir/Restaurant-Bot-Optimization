from restaurantLayout import simulate
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from itertools import zip_longest

# Run the Experiment for a specific number of times
def runSetOfExperiments(duration, numberOfRuns,numberOfBots, method):
    waitingTimeList = [] # List to record the average waiting times
    # Run the experiment multiple times(equal to number of Runs)
    for _ in range(numberOfRuns):
        waitingTimeList.append(runOneExperiment(duration, numberOfBots, method))
    return [sum(filter(None, x)) / len(list(filter(None, x))) for x in zip_longest(waitingTimeList[0], waitingTimeList[1], waitingTimeList[2], waitingTimeList[3], waitingTimeList[4])]
    
# Run the experiment for all parameter combinations
def runExperimentsWithDifferentParameters(duration, maxNoOfBots,numberOfRuns, method):
    resultsTable = {}
    # number of bots from 1 to maxNoOfBots
    for numberOfBots in range(1,maxNoOfBots + 1):
        waitingTimeList = runSetOfExperiments(duration, numberOfRuns,numberOfBots, method)
        resultsTable["robots: "+str(numberOfBots)] = waitingTimeList # Record the results in the resultsTable
    # Determine the maximum length among all arrays in resultsTable
    max_length = max(len(lst) for lst in resultsTable.values())

    # Iterate through each array in resultsTable and append None values to make them equal length
    for key, value in resultsTable.items():
        resultsTable[key] = value + [None] * (max_length - len(value))
    # Convert the results into a dataframe
    results = pd.DataFrame(resultsTable)
    #Store the results in an excel file
    results.to_excel("data/" + method +  "_" + str(duration) + "_" + str(maxNoOfBots) + "_" + str(numberOfRuns) + "_ALL_VALUE_DATA.xlsx")
    # print(ttest_ind(results["robots: 1"],results["robots: 2"]))
  
    # Plot the graph
    ax = results.plot(color=['red', 'green', 'blue', 'orange', 'purple'], legend=True)

    # Set x-axis label
    ax.set_xlabel('Customer Number')

    # Set y-axis label
    ax.set_ylabel('Waiting Time')

    if method == 'astar':
        title = "A* Search"
    elif method == 'dijkstra':
        title = "Dijkstra's Algorithm"
    elif method == 'dfs':
        title = "Depth First Search"
    elif method == 'bfs':
        title = "Breadth First Search"
    ax.set_title(title)

    # Save the boxplot image
    plt.savefig("plots/" + method +  "_" + str(duration) + "_" + str(maxNoOfBots) + "_" + str(numberOfRuns) + "_ALL_VALUE_PLOT.png")
    plt.close()

def runOneExperiment(duration, numberOfBots, method):
    return simulate(duration, numberOfBots, method)


duration = 300
maxNoOfBots = 5
numberOfRuns = 5
methods = [
    "astar", 
    "dijkstra", 
    "bfs", 
    "dfs"
    ]

for method in methods:
    runExperimentsWithDifferentParameters(duration, maxNoOfBots, numberOfRuns, method)


