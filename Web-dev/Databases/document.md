Document Databases (MongoDB)

2. Document Databases (MongoDB)
Example:
* MongoDB
MongoDB stores flexible JSON-like documents.
Instead of rows/columns:

{
  "name": "Alice",
  "age": 21,
  "hobbies": ["guitar", "games"]
}

Another user can have different fields entirely.

Why MongoDB became popular
Web apps evolved quickly.
Frontend data naturally looked like JSON already.
MongoDB let developers move fast without strict schemas.
Good for:
* rapid prototyping
* flexible app data
* user profiles
* content management

SQL vs MongoDB intuition
SQL
Rigid structure:
“every row must look similar”
MongoDB
Flexible:
“store whatever JSON you want”

But MongoDB has weaknesses too
Relationships become messy.
Suppose:
* millions of users
* millions of followers
* recommendation systems
Complex relational queries become difficult.
SQL actually handles many relationship queries better.
So:
MongoDB is not “newer and better SQL”
It’s optimized differently.
