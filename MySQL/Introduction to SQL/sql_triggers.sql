

create table trigger_test(
 message varchar(40)
);


delimiter ##

create 
 trigger trigger1 before insert
on employee
for each row begin
	insert into trigger_test values('new employee added')
end##

delimiter ;
