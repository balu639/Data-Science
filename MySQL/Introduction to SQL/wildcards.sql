-- % = any # characters before that, _= only one character before that

-- find any client's who are an LLC

select *
from client
where client_name LIKE '%LLC';

-- find any branch suppliers who are in the label business
select *
from branch_supplier
where supplier_name LIKE '% Label%';

-- find an employee born in october
select *
from employee
where birth_day like '____-02%';


-- find any client who are schools
select *
from client
where client_name like '%school%';