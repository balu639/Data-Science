-- to get the information from the database management system, we use select statement

select * 
from student;

-- get the specific columns that you need from select statement

select student_id, name 
from student;

-- we can also specify the table name in select query column names just to be clear as in where the data is coming from

select student.student_id, student.name
from student;

-- we want to get those records in an order

select student.student_id, student.name
from student order by student.name desc;

select student.name, student.major
from student order by student.student_id asc;

-- ordering the rows by two columns

select * 
from student
order by major , student_id asc; 

-- limit the results you get

select * 
from student 
limit 2;

select * 
from student
order by major , student_id asc
limit 3;

-- filter and get the information what you need

select *
from student 
where major = 'Bio Chemistry' and name = 'Rose';

/*
Comparison operators.

= --> equals to
< --> Less than
> --> Greater than
<= --> Less than or equal to
>= --> greater than or equal to
<> --> Not equals to
AND, OR

*/
select * 
from student;

select *
from student 
where student.student_id = 5;

select *
from student 
where student.student_id <= 5;

select *
from student 
where student.student_id >= 5;

select *
from student 
where student.major <> 'Bio Chemistry';


select *
from student 
where student.student_id <= 3 and student.major <> 'Bio Chemistry';

-- Using IN operator to give a group of values to specific column to filter the data bases table

select * 
from student;

select * 
from student 
where student.name in ('jame', 'Colon', 'Jack') and student.student_id >= 3;
