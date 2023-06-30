-- find all the employees

select
* from employee;

-- find all clients

select *
from client;

-- Find all the employees ordered by salary

select *
from employee
order by salary desc;

-- find all the employees order by sex and name

select *
from employee 
order by sex, first_name, last_name;

-- find first 5 employees from the employee table

select *
from employee
limit 5;

-- find the first, last names of the employees
select first_name, last_name from employee;

-- find the forename, surname of all the employees
select first_name as forename, last_name as surname from employee;

-- find out all the different genders from employee table

select distinct sex 
from employee;