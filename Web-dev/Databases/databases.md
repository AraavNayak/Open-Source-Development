Databases

Simply put, a system for storing and retrieving data efficiently.


—
Instead of: users = []

in RAM, databases:
* persist data to disk
* support huge scale
* allow searching/filtering
* handle many users simultaneously
* avoid corruption
——


The Big Categories
Most modern databases fall into a few families:
<u>Type |	Good at</u>
Relational (SQL) |	structured/tabular data
Document (MongoDB) |	flexible JSON-like data
Key-value |	ultra-fast simple lookups
Graph DB |	relationships/connections
Time-series |	time-based data
Vector DB |	semantic AI embeddings


Big Picture
Each database type optimizes for something.
Database	Optimized For
SQL	correctness + structure
MongoDB	flexibility
Redis	speed
Graph DB	relationships
Time-series	temporal data
Vector DB	semantic similarity
Modern systems often combine MANY databases
Very important insight:
large systems are polyglot
Example modern AI app:
Need	Database
users/payments	PostgreSQL
caching	Redis
embeddings	Vector DB
logs	Time-series DB
relationships	Graph DB














