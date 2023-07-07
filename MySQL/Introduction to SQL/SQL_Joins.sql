select * from branch;
-- Find all the branches and names of their managers
insert into branch values(4, 'Buffalo', null,null);

select employee.emp_id, employee.first_name, branch.branch_name
from employee
join branch
on employee.emp_id = branch.mgr_id;

-- Left Join: Returns all the rows from the left table(from is used)
select employee.emp_id, employee.first_name, branch.branch_name
from employee
left join branch
on employee.emp_id = branch.mgr_id;

-- Right Join: Returns all the right table rows no matter what(Join is used).
select employee.emp_id, employee.first_name, branch.branch_name
from employee
right join branch
on employee.emp_id = branch.mgr_id;


