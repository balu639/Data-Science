-- Before creating a table, first select the database in which your table needs to be created.

USE mydatabase;
/*
Creating a student table with three columns. Such as student_id,
name and major. Where student_id is the Primary Key for student database.
*/
CREATE TABLE student(
	student_id INT PRIMARY KEY,
    name varchar(20),
    major varchar(20)
);

-- to get the details of your table like the columns it has, their respective datatypes, primary keys and foreign keys, use describe.
describe student;

-- Delete the database table.

DROP table student;

/*
 Modifying a table after it has been created.
*/
ALTER TABLE student ADD gpa DECIMAL(3,2);