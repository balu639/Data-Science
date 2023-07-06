-- find a list of employee and branch names

select first_name as company_name
from employee
union
select branch_name
from branch
union
select client_name
from client;

-- Find a list of all clients & branch suppliers names

select client_name,client.branch_id
from client
union
select supplier_name,branch_supplier.branch_id
from branch_supplier;

-- Find a list of all money spent or earned by the companies

select salary
from employee
union
select total_sales
from works_with;

-- while using union, the column count in all the select statements should be same.
-- Data types of the columns should be same.