from restaurantLayout import simulate
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Run the Experiment for a specific number of times
def runSetOfExperiments(duration, numberOfRuns,numberOfBots, method):
    waitingTimeList = [] # List to record the average waiting times
    # Run the experiment multiple times(equal to number of Runs)
    for _ in range(numberOfRuns):
        waitingTimeList.append(runOneExperiment(duration, numberOfBots, method))
    return waitingTimeList
        
# Run the experiment for all parameter combinations
def runExperimentsWithDifferentParameters(duration, maxNoOfBots,numberOfRuns, method):
    resultsTable = {}
    # number of bots from 1 to maxNoOfBots
    for numberOfBots in range(1,maxNoOfBots + 1):
        waitingTimeList = runSetOfExperiments(duration, numberOfRuns,numberOfBots, method)
        resultsTable["robots: "+str(numberOfBots)] = waitingTimeList # Record the results in the resultsTable
    # Convert the results into a dataframe
    results = pd.DataFrame(resultsTable)
    # Print the results
    print(results)
    #Store the results in an excel file
    results.to_excel("data/" + method +  "_" + str(duration) + "_" + str(maxNoOfBots) + "_" + str(numberOfRuns) + "_data.xlsx")
    # Create boxplots for the results
    boxplot = results.boxplot(grid=False)
    # Set the titles for the boxplot
    boxplot.set_xlabel('Number of Robots')
    boxplot.set_ylabel('Customer Waiting Time (loops)')
    if method == 'astar':
        title = "A* Search"
    elif method == 'dijkstra':
        title = "Dijkstra's Algorithm"
    elif method == 'dfs':
        title = "Depth First Search"
    elif method == 'bfs':
        title = "Breadth First Search"
    boxplot.set_title(title)

    # Save the boxplot image
    plt.savefig("plots/" + method +  "_" + str(duration) + "_" + str(maxNoOfBots) + "_" + str(numberOfRuns) + "_plot.png")
    plt.close()

def runOneExperiment(duration, numberOfBots, method):
    waitingTimes = simulate(duration, numberOfBots, method)
    if len(waitingTimes) == 0:
        return 0
    return sum(waitingTimes)/len(waitingTimes)

duration = 300
maxNoOfBots = 5
numberOfRuns = 10
methods = [
    "astar", 
    "dijkstra", 
    "bfs", 
    "dfs"
    ]

for method in methods:
    runExperimentsWithDifferentParameters(duration, maxNoOfBots, numberOfRuns, method)


