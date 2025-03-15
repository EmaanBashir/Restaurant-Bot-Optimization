**Presentation Link:** https://www.canva.com/design/DAGFHJKYZ_M/Qs39OW4gnlswmK5htBSp6g/edit?utm_content=DAGFHJKYZ_M&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# Restaurant Operations Optimization with Autonomous Bots

## Introduction
This project aims to enhance restaurant operations by simulating an environment where autonomous bots deliver orders to customers. The goal is to analyze how the number of bots and different pathfinding algorithms impact customer waiting times, ultimately improving customer satisfaction and operational efficiency.

## Research Question
How do the number of bots and the choice of pathfinding algorithm impact the average customer waiting time in a restaurant environment?

## Related Work
- **Food Delivery Automation in Restaurants Using Collaborative Robotics** [1]: This paper proposes a centralized system for coordinating robots in a restaurant to streamline food serving processes, aligning with our focus on efficient resource allocation and pathfinding algorithms.
- **Items-mapping and Route Optimization in a Grocery Store** [2]: This study explores optimal shopping routes using various algorithms, emphasizing the importance of minimizing traversal times to enhance customer experiences.
- **Optimal and Efficient Path Planning for Partially-Known Environments** [3]: This paper introduces an algorithm for efficient path planning in dynamic environments, resonating with our exploration of adaptive pathfinding in a restaurant setting.

## Technologies Used and Challenges
- **Tkinter**: Used for implementing the restaurant layout.
- **Matplotlib**: Used for plotting graphs and charts for analysis.
- **Challenges**: Optimizing robot movement, handling interactions between bots and customers, and ensuring smooth GUI responsiveness.

## Environment and Agent Design
- **Simulation Environment**: Includes passive objects like tables, kitchen, doors, and a charging station, along with agents such as customers, robots, and a manager.
- **Restaurant Layout**: Represented by a 1160*760 pixels window with designated areas for the kitchen, charger, doors, and tables.
- **Agents**:
  - **Robots**: Responsible for delivering food, with states indicating battery level and task status.
  - **Customers**: Arrive in groups, select tables, place orders, and leave after receiving their food.
  - **Manager**: Assigns orders to robots based on their availability and proximity to the kitchen.

## System Design
- **Obstacle Avoidance**: Implemented for both customers and robots to navigate around obstacles and other agents.
- **Path Navigation**: Utilizes Depth First Search (DFS), Breadth First Search (BFS), Dijkstra’s Algorithm, and A* Algorithm to find optimal paths for robots.

## Experiment Setup
- **Simulation**: Run for 300 seconds with varying numbers of bots and pathfinding algorithms.
- **Data Collection**: Average customer waiting times recorded and analyzed using box plots.
- **Parameters**: Number of bots varied from 1 to 5 for each algorithm.

## Experiment Results and Discussion
- **Customer Waiting Times**: Analyzed for each algorithm, showing a decrease in waiting times with an increase in the number of robots.
- **Effect of the Number of Robots**: Increasing the number of robots reduces waiting times, but the impact diminishes beyond a certain threshold.
- **Effect of the Pathfinding Algorithm**: A* Algorithm was found to be the most optimal, providing the shortest paths and minimizing waiting times.

## Conclusion
The project demonstrates that a combination of an optimal number of robots and an efficient pathfinding algorithm can significantly enhance restaurant operations by minimizing customer waiting times. Further research could explore additional factors such as restaurant layout optimization and dynamic adaptation of algorithms to real-time conditions.

## Future Work
- **Expand Robot Tasks**: Include tasks like cleaning and guiding customers.
- **Improve Pathfinding Algorithms**: Explore other algorithms to further optimize robot navigation.

## References
- [1] Antony, A., & Sivraj, P. (2018). Food Delivery Automation in Restaurants Using Collaborative Robotics.
- [2] Dela Cruz, J. C., et al. (2016). Items-mapping and Route Optimization in a Grocery Store.
- [3] Stentz, A. (1994). Optimal and Efficient Path Planning for Partially-Known Environments.
