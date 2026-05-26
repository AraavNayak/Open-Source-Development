5. Time-Series Databases
Examples:
* InfluxDB
* TimescaleDB
Optimized for:
data over time
Example:
timestamp	CPU
1:00	40%
1:01	45%
1:02	90%
Used for:
* sensors
* monitoring
* stock data
* observability

Why normal DBs struggle here
Time-series data:
* huge
* append-heavy
* queried by ranges
Specialized systems compress/store this better.
