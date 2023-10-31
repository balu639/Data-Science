-- Find the count of the employees
select count(emp_id)
from employee;


select * from employee;
-- find the employee count who has supervisor id

select count(supervisor_id)
from employee;

-- find the number of female employees born after 1970

select count(emp_id)
from employee
where sex = 'F' and birth_day > '1970-01-01';

-- find the average salary of all the employees

select avg(salary)
from employee;

-- Find the average salary of all the male employees
select avg(salary)
from employee
where sex = 'M';

-- Find the sum of all the employees
select sum(salary)
from employee;

-- Find how many males and females in the employee table
select count(sex), sex
from employee
group by sex;

-- Find each employee total sales
select emp_id, sum(total_sales)
from works_with
group by emp_id;

-- Find how much money each client spent

select client_id , SUM(total_sales)
from works_with
group by client_id;