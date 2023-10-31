## MySQL
MySQL is an open-source relational database management system (RDBMS) that is widely used for managing and storing data. It is a popular choice for web applications and is used by many organizations, ranging from small businesses to large enterprises.

### SQL Basics

#### Create
This operation is used to insert new records into a database table. It involves specifying the data to be inserted and providing the necessary values for the fields or columns in the table.

                                CREATE TABLE student(
                              	student_id INT PRIMARY KEY,
                                  name varchar(20),
                                  major varchar(20)
                              ); 

                              INSERT INTO student VALUES(1, 'John', 'Computer SCience'); 

#### Read
The read operation is used to retrieve data from the database. It involves querying the database table and fetching the desired records based on certain conditions. Read operations can be simple queries to retrieve specific records or complex queries involving joins, aggregations, and sorting.

                              select * 
                              from student; 

                              select student_id, name 
                                from student; 

#### Update
The update operation is used to modify existing records in the database. It involves selecting the records to be updated and specifying the new values for one or more fields. Update operations allow you to change specific data within a record without deleting and re-creating it.

                              update employee
                              set branch_id =1
                              where emp_id =100; 

#### Delete
The delete operation is used to remove records from the database table. It involves selecting the records to be deleted based on certain conditions and removing them from the table. Delete operations permanently remove the data from the database.

                              DELETE FROM customers
                                WHERE id = 5;

                              DELETE TABLE student;


### SQL Joins

In SQL, a join is a method used to combine rows from two or more tables based on related columns between them. It allows you to retrieve data from multiple tables in a single query by specifying the conditions for joining the tables.

There are different types of SQL joins that you can use, including:

#### Inner Join

Returns only the rows that have matching values in both tables being joined.

                          select employee.emp_id, employee.first_name, branch.branch_name
                          from employee
                          inner join branch
                          on employee.emp_id = branch.mgr_id;

#### Left Join
Returns all the rows from the left table and the matched rows from the right table. If there are no matches, NULL values are returned for the right table columns.

                        select employee.emp_id, employee.first_name, branch.branch_name
                        from employee
                        left join branch
                        on employee.emp_id = branch.mgr_id;

#### Right Join
Returns all the rows from the right table and the matched rows from the left table. If there are no matches, NULL values are returned for the left table columns.

                        select employee.emp_id, employee.first_name, branch.branch_name
                        from employee
                        right join branch
                        on employee.emp_id = branch.mgr_id;

#### Full Join
Returns all the rows from both tables. If there are no matches, NULL values are returned for columns from the table that doesn't have a match.
                        
                        select employee.emp_id, employee.first_name, branch.branch_name
                        from employee
                        join branch
                        on employee.emp_id = branch.mgr_id;
                      

### SQL Union

SQL UNION is a keyword used to combine the result sets of two or more SELECT statements into a single result set. The UNION operation removes duplicate rows from the combined result set.

The basic syntax for using UNION is as follows:

                      select first_name as company_name
                      from employee
                      union
                      select branch_name
                      from branch
                      union
                      select client_name
                      from client;

The UNION operation combines the result sets of the two SELECT statements and removes duplicate rows. The result set is sorted in ascending order by default. If you want to sort the result set in a specific order, you can use the ORDER BY clause at the end of the query.


                      select client_name,client.branch_id
                      from client
                      union
                      select supplier_name,branch_supplier.branch_id
                      from branch_supplier
                      order by client_name;


### Nested Queries
Nested queries, also known as subqueries, are SQL queries that are embedded within another query. A nested query is used to retrieve data from one or more tables based on the results of an outer query. It allows you to perform complex operations and make the query more flexible by using the results of one query as input for another.

There are two types of nested queries: correlated and non-correlated.

1. Non-correlated:

A non-correlated nested query is executed independently of the outer query and doesn't reference any columns from the outer query. It is executed first, and its result is used in the outer query.

Here's an example of a non-correlated nested query:
                    
                    SELECT name, salary
                    FROM employees
                    WHERE salary > (SELECT AVG(salary) FROM employees);

2. correlated nested queries:
A correlated nested query is executed for each row of the outer query. It references columns from the outer query, enabling the nested query to be dependent on the outer query's values.

Here's an example of a correlated nested query:
                    
                    SELECT name, salary
                    FROM employees e
                    WHERE salary > (SELECT AVG(salary) FROM employees WHERE department = e.department);




### SQL Functions
SQL functions are built-in operations that perform specific tasks or calculations on data in a database. They can be used to manipulate, transform, or retrieve data in various ways. SQL functions can be categorized into several types:

Few examples of functions:
1. COUNT
2. SUM
3. AVG

### SQL triggers
In SQL, triggers are database objects that are automatically executed in response to specific events, such as data modifications (INSERT, UPDATE, DELETE) or certain actions (e.g., login or logout). Triggers are typically used to enforce business rules, maintain data integrity, and automate certain tasks within the database.

                    CREATE TRIGGER after_insert_example
                    AFTER INSERT
                    ON employees
                    FOR EACH ROW
                    BEGIN
                        INSERT INTO audit_table (event_type, event_date, user_id)
                        VALUES ('INSERT', NOW(), NEW.user_id);
                    END;


### On Delete
In SQL, the ON DELETE clause is used to specify the action to be taken when a row in a parent table is deleted and there are related rows in the child table(s). It is typically used in foreign key relationships to define the referential integrity between tables.

ON DELETE SET NULL: When a row in the parent table is deleted, all related rows in the child table(s) will have their foreign key columns set to NULL. This is applicable only if the foreign key columns allow NULL values.

                  create table branch (
                  branch_id INT PRIMARY KEY,
                  branch_name VARCHAR(40),
                  mgr_id INT,
                  mgr_start_date DATE,
                  foreign key(mgr_id) references employee(emp_id) on delete set null
                  );


ON DELETE CASCADE: When a row in the parent table is deleted, all related rows in the child table(s) will also be deleted automatically. This option effectively propagates the delete operation from the parent to the child tables.

                  
                  create table works_with(
                  emp_id INT,
                  client_id INT,
                  total_sales INT,
                  primary key(emp_id,client_id),
                  foreign key (emp_id) references employee(emp_id) on delete cascade,
                  foreign key (client_id) references client(client_id) on delete cascade
                  );

### Wild Cards

SQL wildcards are special characters used in SQL (Structured Query Language) to perform pattern-matching searches in queries. They allow you to search for data that matches a specific pattern rather than an exact value. The three main SQL wildcards are:

1. %(percent sign):
The percent sign represents zero, one, or multiple characters. It can be used to match any sequence of characters in a string. For example, if you use the pattern 'A%', it will match all values that start with the letter 'A'.

                  select *
                  from client
                  where client_name LIKE '%LLC';

3. '__'(underscore):_
The underscore represents a single character. It can be used to match a single character at a specific position in a string.For example, if you use the pattern 'J__n', it will match values like 'John', 'Jane', etc.,

                   select *
                  from employee
                  where birth_day like '____-02%';

5. '[]'(square brackets):
Square brackets allow you to specify a character range to match a single character. You can define a set of characters enclosed within square brackets to match any one of those characters. For example, if you use the pattern 'M[ae]%', it will match values that start with 'M' followed by either 'a' or 'e'.
