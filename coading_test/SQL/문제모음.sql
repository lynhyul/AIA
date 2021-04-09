-- 아이디 역순 정렬하기
SELECT NAME, DATETIME
FROM ANIMAL_INS
ORDER BY ANIMAL_ID DESC

-- 아픈 동물 찾기

SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
WHERE INTAKE_CONDITION LIKE 'Sick'

-- 젊은 동물 찾기

SELECT ANIMAL_ID, name
FROM ANIMAL_INS
WHERE INTAKE_CONDITION NOT LIKE 'Aged'


-- 동물 아이디와 이름
SELECT ANIMAL_ID, name
FROM ANIMAL_INS
ORDER BY ANIMAL_ID

-- 여러 기준으로 정렬하기 (날짜만 역순)
SELECT ANIMAL_ID, NAME, DATETIME
FROM ANIMAL_INS
ORDER BY NAME,DATETIME DESC

-- 상위 N개의 레코드
SELECT NAME
FROM ANIMAL_INS
ORDER BY DATETIME
LIMIT 1

-- 최댓값
SELECT MAX(DATETIME) AS '시간'
FROM ANIMAL_INS

-- #동물 수 구하기
SELECT COUNT(*)
FROM ANIMAL_INS



-- #중복 제거하기
-- 일단 서브쿼리를 사용하는데
-- NAME이 NULL이 아니고 NAME별로 GROUP을 해준 후 ANIMAL_INS 테이블의 NAME들만 조회하는데
-- 이후 밖에서는 이 행들의 갯수만 COUNT(*)로 세주면 바로 끝이 난다
-- 서브쿼리를 사용할 때 반드시 서브쿼리의 이름을 정해줘야 에러가 발생하지 않는다

SELECT COUNT(*)
FROM
    (
    SELECT NAME
    FROM ANIMAL_INS
    WHERE NAME IS NOT NULL
    GROUP BY NAME
    ) SQ1



-- #개와 고양이는 몇 마리 있을까?(고양이 순서 우선)
-- GROUP BY란?
-- 컬럼 값이 같은 것 끼리 하나로 묶어준다는 거다
-- ANIMAL_TYPE으로 묶어놓으니
-- Cat과 Dog이 몇마린지 정리된 것이 보인다
-- ORDER BY CASE로 다음과 같이 순서도 정의 할 수 있다.

SELECT ANIMAL_TYPE,COUNT(ANIMAL_TYPE)
FROM ANIMAL_INS
GROUP BY ANIMAL_TYPE
ORDER BY
CASE ANIMAL_TYPE
	WHEN 'Cat' THEN 1
	WHEN 'Dog' THEN 2
	ELSE 3
END


-- 동명 동물 수 찾기
SELECT NAME, COUNT(*) AS COUNT 
FROM ANIMAL_INS 
GROUP BY NAME 
HAVING COUNT(NAME) >= 2 
ORDER BY NAME

-- 입양 시간 구하기(9~ 19시까지만)
SELECT HOUR(DATETIME) AS HOUR, COUNT(DATETIME) AS COUNT 
FROM ANIMAL_OUTS 
WHERE HOUR(DATETIME) >= 9 AND HOUR(DATETIME) <= 19
GROUP BY HOUR
ORDER BY HOUR

-- 입양 시간 구하기(0시 부터 표시하기)
-- 변수 지정하는것을 잘 기억해두자
SET @hour = -1; 
SELECT @hour := @hour + 1 AS HOUR, (
    SELECT COUNT(DATETIME)
    FROM ANIMAL_OUTS B 
    WHERE HOUR(DATETIME) = @hour ) AS COUNT 
FROM ANIMAL_OUTS A 
WHERE @hour < 23

-- DATETIME에서 DATE로 형 변환
SELECT ANIMAL_ID, NAME, date_format(datetime, '%Y-%m-%d')
FROM ANIMAL_INS
ORDER BY ANIMAL_ID

-- 오랜 기간 보호한 동물

SELECT
AIN.NAME,
AIN.DATETIME
FROM
ANIMAL_INS AIN
LEFT JOIN
ANIMAL_OUTS ATO
ON
AIN.ANIMAL_ID = ATO.ANIMAL_ID
WHERE
ATO.ANIMAL_ID IS NULL
ORDER BY AIN.DATETIME ASC LIMIT 3

-- 보호소에서 중성화한 동물
SELECT INS.ANIMAL_ID, INS.ANIMAL_TYPE, INS.NAME
FROM ANIMAL_INS INS
INNER JOIN ANIMAL_OUTS OUTS
ON INS.ANIMAL_ID = OUTS.ANIMAL_ID AND INS.SEX_UPON_INTAKE != OUTS.SEX_UPON_OUTCOME; -- A값이 B값과 다를 경우 true , 같을경우 fals 반환


-- 루시와 엘라 찾기
-- 동물 보호소에 들어온 동물 중 이름이 Lucy, Ella, Pickle, Rogan, Sabrina, Mitty인 
-- 동물의 아이디와 이름, 성별 및 중성화 여부를 조회하는 SQL 문을 작성해주세요.
SELECT animal_id, name, sex_upon_intake
FROM ANIMAL_INS
WHERE name in ('Lucy','Ella','Pickle', 'Rogan', 'Sabrina', 'Mitty')
ORDER BY animal_id

-- 이름에 el 들어가는 아이 찾기
-- 코드를 입력하세요
SELECT ANIMAL_ID, NAME 
FROM ANIMAL_INS 
WHERE NAME LIKE '%EL%' AND ANIMAL_TYPE = 'Dog' 
ORDER BY NAME

-- 중성화 여부 파악하기
-- 코드를 입력하세요
-- 보호소의 동물이 중성화되었는지 아닌지 파악하려 합니다. 중성화된 동물은 SEX_UPON_INTAKE 컬럼에 'Neutered' 또는 'Spayed'라는 단어가 들어있습니다. 
-- 동물의 아이디와 이름, 중성화 여부를 아이디 순으로 조회하는 SQL문을 작성해주세요. 이때 중성화가 되어있다면 'O', 아니라면 'X'라고 표시해주세요
SELECT animal_id, name, if(sex_upon_intake like 'Intact%', 'X', 'O') 
from animal_ins
order by animal_id;

-- 오랜기간 보호한 동물
-- 코드를 입력하세요
-- 입양을 간 동물 중, 보호 기간이 가장 길었던 동물 
-- 두 마리의 아이디와 이름을 조회하는 SQL문을 작성해주세요. 이때 결과는 보호 기간이 긴 순으로 조회해야 합니다.
SELECT A.ANIMAL_ID, A.NAME 
FROM ANIMAL_INS A, ANIMAL_OUTS B 
WHERE A.ANIMAL_ID = B.ANIMAL_ID 
ORDER BY DATEDIFF(A.DATETIME, B.DATETIME)  -- DATEDIFF() 함수는 두개의 날짜값의 차이를 int로 반환하는 Mssql 내장함수이다
LIMIT 2


-- 없어진 기록
-- 천재지변으로 인해 일부 데이터가 유실되었습니다. 입양을 간 기록은 있는데, 
-- 보호소에 들어온 기록이 없는 동물의 ID와 이름을 ID 순으로 조회하는 SQL문을 작성해주세요.
-- 코드를 입력하세요
SELECT OUTS.ANIMAL_ID, OUTS.NAME
FROM ANIMAL_OUTS OUTS
LEFT JOIN ANIMAL_INS INS
ON OUTS.ANIMAL_ID = INS.ANIMAL_ID
WHERE INS.ANIMAL_ID is NULL
ORDER BY OUTS.ANIMAL_ID


-- 있었는데요 없었습니다.
-- 관리자의 실수로 일부 동물의 입양일이 잘못 입력되었습니다. 
-- 보호 시작일보다 입양일이 더 빠른 동물의 아이디와 이름을 조회하는 SQL문을 작성해주세요. 
-- 이때 결과는 보호 시작일이 빠른 순으로 조회해야합니다.

SELECT INS.ANIMAL_ID, INS.NAME
FROM ANIMAL_INS INS
JOIN ANIMAL_OUTS OUTS
ON INS.ANIMAL_ID = OUTS.ANIMAL_ID
WHERE INS.DATETIME > OUTS.DATETIME
ORDER BY INS.DATETIME