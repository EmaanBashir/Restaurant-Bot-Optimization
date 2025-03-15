# 🍽️ Restaurant Bot Simulation: Optimizing Food Delivery with Pathfinding Algorithms

![restaurant_simulation](https://github.com/user-attachments/assets/0ad983a2-d1c5-423b-bbc3-c9cf6838b4b0)

## 📌 Project Overview  
This project simulates a restaurant environment to analyze the impact of the number of autonomous delivery bots and different pathfinding algorithms on customer waiting times. The goal is to determine the optimal number of robots and the most efficient pathfinding algorithm for minimizing delivery time in a restaurant setting.  

## ❓ Research Question  
**How do the number of bots and the choice of pathfinding algorithm impact the average customer waiting time in a restaurant environment?**  

## 🔬 Technologies Used  
- 🐍 **Python** (Simulation & Algorithm Implementation)  
- 🖥 **Tkinter** (GUI for the Restaurant Layout)  
- 📊 **Matplotlib** (Data Visualization & Graphs)  
- 📑 **Excel** (Result Storage & Analysis)  

## 🏗 System Design  
The restaurant environment consists of passive objects (tables, kitchen, doors, charging station) and agents (customers, robots, and a manager).  

### 📌 Restaurant Layout
![restaurant_layout](https://github.com/user-attachments/assets/56cbac4b-0b05-40b5-b1ab-7f02dbdd4f97)

- **🤖 Robots**: Deliver food from the kitchen to customer tables using different pathfinding algorithms.  
- **👥 Customers**: Arrive in groups, occupy tables, place orders, and leave after eating.  
- **👨‍💼 Manager**: Assigns food delivery tasks to available robots based on efficiency criteria.  

### 🏃‍♂️ Robot States  
![robot_states](https://github.com/user-attachments/assets/1bf40797-4323-4e57-9233-bdcb91f5c69d)

## 🛤 Pathfinding Algorithms Implemented  
1. **🔍 Depth First Search (DFS)**: Explores paths exhaustively, often resulting in long and inefficient routes.  
2. **🌳 Breadth First Search (BFS)**: Finds the shortest path but can be computationally expensive.  
3. **📍 Dijkstra’s Algorithm**: Optimizes shortest path calculations based on cost but has more turns.  
4. **⭐ A* Algorithm**: Combines cost and heuristic measures for the most efficient pathfinding, making it the best-performing algorithm in this study.  

### 📍 Path Comparisons  
![pathfinding_algorithms](https://github.com/user-attachments/assets/158f5f21-b9b4-480e-b66d-1b731255afb9)

## 📊 Experiment Setup  
- The simulation runs for **300 seconds** per test.  
- **Number of Bots**: Varied from **1 to 5**.  
- **Pathfinding Algorithms**: Each algorithm was tested across different bot counts.  
- **Performance Metrics**: Average customer waiting times were recorded and analyzed using box plots.  

## 📈 Experiment Results  

### 📉 Effect of the Number of Robots on Waiting Time  
![waiting_time_vs_robots](https://github.com/user-attachments/assets/e1eeb3ac-7aad-4c24-8c7c-99a34d03d69e)

- **Increasing the number of robots reduces customer waiting time** but reaches a saturation point where additional bots provide diminishing returns.  

### 📉 Effect of Pathfinding Algorithm on Waiting Time  
![waiting_time_vs_algorithm](https://github.com/user-attachments/assets/57de2fbc-5853-4711-9937-fe03f708d0c3)

- **A* Algorithm performed the best**, minimizing customer waiting times compared to other algorithms.  
- **Depth First Search was the least efficient**, leading to significantly longer delivery times.  

## 🏆 Key Findings  
✅ **Optimal number of robots = 3** (Adding more bots does not significantly improve waiting times).  
✅ **A* Algorithm is the most efficient** for minimizing customer waiting time.  
✅ **DFS is highly inefficient** and not suitable for real-time applications.  

## 📌 Future Work  
- Expanding robot functionalities to include **cleaning and customer guidance**.  
- Exploring additional **pathfinding algorithms** for further optimization.  
- Adapting the system for **real-world restaurant layouts** with dynamic obstacles.

## 🚀 How to Run the Code  

### 📦 Requirements  
Ensure you have Python installed and install the required dependencies using:  
```bash  
pip install matplotlib tkinter  
```
### Clone the repository:
```sh
git clone https://github.com/EmaanBashir/Restaurant-Bot-Optimization.git
cd Restaurant-Bot-Optimization/Code
```

### 🏨 Running the Restaurant Simulation  
To launch the restaurant environment with bots and customer interactions, run:  
```bash  
python restaurantLayout.py  
```

### 📊 Running Experiments  
To analyze customer waiting times based on different parameters, run the respective scripts:  

#### 🤖 Number of Robots vs Waiting Time  
Run the following command to analyze the effect of the number of robots on waiting time:  
```bash  
python runAllExperiments(NoOfRobotsvsWaitingTime).py  
```

#### 🏛️ Customer Number vs Waiting Time  
Run the following command to analyze the effect of customer numbers on waiting time:  
```bash  
python runAllExperiments(CustomerNovsWaitingTime).py  
```

## Presentation Link
https://www.canva.com/design/DAGFHJKYZ_M/Qs39OW4gnlswmK5htBSp6g/edit?utm_content=DAGFHJKYZ_M&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

---
For further details check the full Report.pdf included in the repository
