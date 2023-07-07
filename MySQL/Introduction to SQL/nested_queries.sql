-- find names of all employees who sold more than 30000 to a
-- single client
select * from employee;
select * from works_with;
select employee.first_name, employee.last_name
from employee
where employee.emp_id IN (
    select works_with.emp_id
	from works_with 
	where works_with.total_sales > 30000


);

/*
Find all the clients who are handled by the branch 
that Micheal Scott manages, Assume you know Micheal's ID

*/

select client.client_name
from client
where client.branch_id = (

	select branch.branch_id 
	from branch
	where branch.mgr_id = '102'
	limit 1
);




