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

### SQL Union

### Nested Queries

### SQL Functions

### SQL triggers

### On Delete

### Wild Cards
