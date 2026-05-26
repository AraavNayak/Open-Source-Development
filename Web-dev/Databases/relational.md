Relational Databases (SQL)


Examples:
* PostgreSQL
* MySQL
* SQLite
These are the “classic” databases.

Core idea
Store data in tables.
Example:
Users table
id	name	age
1	Alice	21
2	Bob	19
Orders table
order_id	user_id	item
101	1	Laptop
102	2	Phone
Notice:
* users have IDs
* orders reference users
This is called a relational database because tables relate to each other.

Why SQL databases are amazing
They are VERY good at:
* precise structured data
* consistency
* financial systems
* transactions
Example: bank transfer
You NEVER want:
* money disappearing
* partial updates
SQL databases are built for reliability.


SQL
You query them with SQL:

SELECT * FROM users WHERE age > 20;

or

SELECT users.name, orders.item
FROM users
JOIN orders
ON users.id = orders.user_id;

That JOIN is the magic: you can combine related tables.



Why relational databases became dominant
Because most business data is structured.

Examples:
* banking
* inventory
* payroll
* school systems
* ecommerce
All naturally fit into tables.


