Graph Databases

Examples:
* Neo4j
* TigerGraph
These store:
* nodes
* edges (connections)
Example:

Alice --friends--> Bob
Bob --works_at--> Google
Google --located_in--> California


Why graphs matter
Some problems ARE mostly relationships.
Examples:
* social networks
* fraud detection
* recommendation systems
* dependency analysis

Why SQL struggles here
Suppose:
“find friends of friends of friends who like jazz”
This becomes ugly in SQL.
Graph DBs make traversal natural.

Knowledge Graphs
A knowledge graph is basically:
a graph database with semantic meaning
Example:

Paris --capital_of--> France
France --part_of--> Europe

or

Service A --depends_on--> Kafka
Kafka --caused--> Incident 42

Used heavily in:
* AI
* search
* enterprise systems
