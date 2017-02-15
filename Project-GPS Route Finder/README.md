Inpurt Format:
python route.py [start-city] [end-city] [routing-option] [routing-algorithm]

where:
 start-city and end-city are the cities we need a route between.
 routing-option is one of:
{ segments finds a route with the fewest number of \turns" (i.e. edges of the graph)
{ distance finds a route with the shortest total distance
{ time finds the fastest route, for a car that always travels at the speed limit
{ scenic finds the route having the least possible distance spent on highways (which we define as roads with speed limits 55 mph or greater)
 routing-algorithm is one of:
{ bfs uses breadth-first search
{ dfs uses depth-first search
{ ids uses iterative deepening search
{ astar uses A* search, with a suitable heuristic function

Output Format:
[total-distance-in-miles] [total-time-in-hours] [start-city] [city-1] [city-2] ... [end-city]

Example:
51 1.0795 Bloomington,_Indiana Martinsville,_Indiana Jct_I-465_&_IN_37_S,_Indiana Indianapolis,_Indiana


# Formulation of search problem:
The problem is built into a graph with the nodes as cities and road segments as edges.
Now we have to search a path from the start city to end city based on the additional information given by the user.

Abstraction:
a) State space: All list of cities and junctions constitute the total state space.

b) Successor function: Cities and junctions connected to a particular city/junction are the successors
of that city/junction.

c) Edge weights: Depending on the type of routing option, the edge weights could either be the distance, time,
segment(weight of 1) or scenic(weight of highway distance of a road)

d) Heuristic:
Heuristic function is only used in A*.
Segment and Distance heuristic – Calculate the haversine distance between the current state and goal state.
This is admissible because the haversine distance will give the minimum distance between the two states given that a
straightline road(curve) exists between the two states.

Time: Since time = distance/speed,take the min distance possible and max speed possible to guarantee an underestimate.
Since dataset has max speed limit as 65, 65 is used as the speed limit.

Scenic: Here we check a possible outgoing routes and if any of them is scenic, it will return 0,
else if all are highways then it will take the minimum of the available highway distances.
This is admissible because if there is at least one outgoing scenic route there is a possibility that
future all roads to the destination would be scenic and thus the heuristics is accurate at 0.
If all are highways, then we know that no matter what the minimum value added now to the highway distance
would be the minimum of the available highways to take.


How the algorithm works:
BFS: Keep traversing the nodes in BFS order and print when destination is found.
DFS: Keep traversing the nodes in DFS order and print when destination is found.
IDS: Keep traversing the nodes in IDS manner, increasing the depth by 1 for every iteration
and print when destination is found.
A*: Implemented A* using the heuristics as defined above.

Assumptions/simplifications:
Since latitude and longitude of junctions are not given, we have calculated them by using the formula:
Summation of |total_of_connecting_segments – length_of_each_connecting_segment|  /total_of_connecting_segments
