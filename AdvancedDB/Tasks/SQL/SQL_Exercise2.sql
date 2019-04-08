use [AdvancedDB]
go


-- Q1
select Lake, Depth
	from Lakes, Countries
	where Lakes.Country =Countries.Country
	and Countries.Continent='Europe'


-- Q2
select distinct Continent, Lake
	from Lakes, Countries
	where Lakes.Country = Countries.Country

-- Q3
select Continent, count(distinct Lake) as lake_number
	from Lakes, Countries
	where Lakes.Country = Countries.Country
	group by Continent;

-- Q4
select Continent, count(distinct Lake) as lake_number
	from Lakes, Countries
	where Lakes.Country = Countries.Country
	group by Continent
	having count(distinct Lake) >= 5

-- Q5
Select Lake, count(distinct Country)
	from Lakes
	group by Lake

--Q6
select max(depth) from Lakes;

--Q7
with Max(Max_Depth) as
	(select max(Depth) from Lakes)

	select distinct Lake,Max_Depth
	from Lakes, Max
	where Depth = Max_Depth
	

--Q8
select distinct Lake, Max(Depth)
	From Countries, Lakes
	where Countries.Country = Lakes.Country
	and Continent = 'Africa'
	group by Lake;

--Q9
with Max(Max_Depth) as
	(select max(Depth) from Lakes,Countries
	where Countries.Country = Lakes.Country
	and Continent = 'Africa'
	)

select distinct Lake, Depth
	From Lakes, Max
	where Depth = Max_Depth;


select distinct Lake, Depth
	From Lakes
	where Depth = (select max(Depth) from Lakes,Countries
	where Countries.Country = Lakes.Country
	and Continent = 'Africa')


--Q10
select Lake, Max(Depth)
	from Countries C, Lakes L
	where C.Country = L.Country
	and Continent = 'Africa'
	group by Lake

--Q11
select Lake, avg(Depth)
from Lakes
group by Lake

	
