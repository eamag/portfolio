-- #1
SELECT SQLDATE, Actor1Name, Actor2Name, Count(*) AS Count
FROM [gdelt-bq:full.events]
WHERE (SQLDATE == 20171001) and (Actor1Name != 'null') and (Actor2Name != 'null')
Group by SQLDATE, Actor1Name, Actor2Name
HAVING COUNT(*)>30
ORDER BY Count DESC

--#2

--SELECT ActionGeo_Lat, ActionGeo_Long, GoldsteinScale   FROM [gdelt-bq:full.events] 
--where (ActionGeo_Lat!=null) and (ActionGeo_Long!=null) 
--output is Query returned zero records

-- #3
select a.SQLDATE, a.Actor1Name, a.SOURCEURL, a.GoldsteinScale 
FROM [gdelt-bq:full.events] a
inner join
(
  SELECT SQLDATE, Actor1Name, max(round(GoldsteinScale, 1)) as goldstein
  FROM [gdelt-bq:full.events]
  WHERE (SQLDATE == 20171001) and (Actor1Name != 'null')
  Group by SQLDATE, Actor1Name) b
on a.SQLDATE = b.SQLDATE and a.Actor1Name = b.Actor1Name and a.GoldsteinScale=b.goldstein  
ORDER BY a.Actor1Name Asc  


-- #4
-- country code here is somewhat different from the example, so there are no uk or spain (ES, not SP) and Switzerland (CH) instead of china (CN)
--select *
--from [gdelt-bq:extra.countryinfo] 
--where country == 'Spain'

select b.ActionGeo_CountryCode, b.NumArticles, a.country, a.capital, a.area, a.population, a.currency_name 
FROM [gdelt-bq:extra.countryinfo] a
right OUTER join each
(
select ActionGeo_CountryCode, sum(NumArticles) as NumArticles
  from [gdelt-bq:full.events]
  where ActionGeo_CountryCode != 'null' and SQLDATE = 20171001
  group by ActionGeo_CountryCode
  ) b
on b.ActionGeo_CountryCode = a.iso
order by b.NumArticles desc

--#5
--similar to #2, there are no ActionGeo_Lat. I can use Actor1Geo_Lat and Actor1Geo_Long, but results will be different
--so I think it will be fine with sql task
