DROP TABLE student;

CREATE TABLE student(
	student_id INT PRIMARY KEY,
    name varchar(20) NOT NULL,
    major varchar(20) default 'undecided'
);

select * from student;

insert into students values(1, 'jack', 'Biology');
insert into students values(2, 'Katey', 'Sociology');

insert into student(student_id, name) values(1, 'james');

CREATE TABLE student(
	student_id INT PRIMARY KEY auto_increment,
    name varchar(20) NOT NULL,
    major varchar(20) default 'undecided'
);


insert into student(name, major) values('Jame', 'sociology');
insert into student(name, major) values('Colon', 'Biology');
insert into student(name) values ('Tom cook');
insert into student(name, major) values ('Rose', 'Bio');
Insert into student (name,major) values ('Jack', 'Chemistry');


select * from student;

SET SQL_SAFE_UPDATES = 0;

update student set major = 'Bio' where major = 'Biology';
update student set major = 'Soc' where major = 'sociology';

update student set major = 'Bio Chemistry' where major = 'Bio' or major= 'Chemistry';

update student set name = 'Tom', major ='undecided' where student_id = 3;

-- updates all records major to undecided
update student set major = 'undecided'; 

-- Deleting rows in a database table
delete from student where student_id = 3;

delete from student where name = 'Tom cook' and major = 'undecided';