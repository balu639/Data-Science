-- on delete set null : sets the foreign key to null upon deletion from the main table.
-- on delete cascade: Deletes the entire 

delete from employee
where emp_id = 103;

select * from branch;

delete from branch
where branch_id = 2;

select * from branch_supplier;