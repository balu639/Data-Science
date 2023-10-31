
-- Insert the data into the database column
INSERT INTO student VALUES(1, 'John', 'Computer SCience','3.47')

select * from student;


INSERT INTO student VALUES(2, 'Jack', 'Applied Datascience','3.50')
/*
When we don't have the values for all the columns for a record. we can insert the values as below:
*/

INSERT INTO student(student_id, name, gpa) VALUES(3, 'Rose','3.58')

