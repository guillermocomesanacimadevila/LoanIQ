-- ===========================================
-- STEP 1: Drop rows with critical missing values
-- ===========================================
DELETE FROM loan_data
WHERE Age IS NULL OR Income IS NULL OR CreditScore IS NULL OR LoanAmount IS NULL;

-- ===========================================
-- STEP 2: Impute missing numeric fields
-- ===========================================

-- MonthsEmployed: approximate median using rank
WITH ordered AS (
    SELECT MonthsEmployed
    FROM loan_data
    WHERE MonthsEmployed IS NOT NULL
    ORDER BY MonthsEmployed
),
counted AS (
    SELECT COUNT(*) AS cnt FROM ordered
),
median_val AS (
    SELECT MonthsEmployed
    FROM (
        SELECT MonthsEmployed, ROW_NUMBER() OVER () AS rn
        FROM ordered
    ) AS ranked, counted
    WHERE rn = (cnt / 2) + 1
)
UPDATE loan_data
SET MonthsEmployed = (SELECT MonthsEmployed FROM median_val)
WHERE MonthsEmployed IS NULL;

-- InterestRate: use mean
UPDATE loan_data
SET InterestRate = (
    SELECT AVG(InterestRate)
    FROM loan_data
    WHERE InterestRate IS NOT NULL
)
WHERE InterestRate IS NULL;

-- DTIRatio: use mean
UPDATE loan_data
SET DTIRatio = (
    SELECT AVG(DTIRatio)
    FROM loan_data
    WHERE DTIRatio IS NOT NULL
)
WHERE DTIRatio IS NULL;

-- ===========================================
-- STEP 3: Impute missing categorical values with mode
-- ===========================================

-- Helper: Replace NULL with most frequent (mode) value

-- Education
UPDATE loan_data
SET Education = (
    SELECT Education
    FROM loan_data
    WHERE Education IS NOT NULL
    GROUP BY Education
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE Education IS NULL;

-- EmploymentType
UPDATE loan_data
SET EmploymentType = (
    SELECT EmploymentType
    FROM loan_data
    WHERE EmploymentType IS NOT NULL
    GROUP BY EmploymentType
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE EmploymentType IS NULL;

-- MaritalStatus
UPDATE loan_data
SET MaritalStatus = (
    SELECT MaritalStatus
    FROM loan_data
    WHERE MaritalStatus IS NOT NULL
    GROUP BY MaritalStatus
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE MaritalStatus IS NULL;

-- HasMortgage
UPDATE loan_data
SET HasMortgage = (
    SELECT HasMortgage
    FROM loan_data
    WHERE HasMortgage IS NOT NULL
    GROUP BY HasMortgage
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE HasMortgage IS NULL;

-- HasDependents
UPDATE loan_data
SET HasDependents = (
    SELECT HasDependents
    FROM loan_data
    WHERE HasDependents IS NOT NULL
    GROUP BY HasDependents
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE HasDependents IS NULL;

-- HasCoSigner
UPDATE loan_data
SET HasCoSigner = (
    SELECT HasCoSigner
    FROM loan_data
    WHERE HasCoSigner IS NOT NULL
    GROUP BY HasCoSigner
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE HasCoSigner IS NULL;

-- LoanPurpose
UPDATE loan_data
SET LoanPurpose = (
    SELECT LoanPurpose
    FROM loan_data
    WHERE LoanPurpose IS NOT NULL
    GROUP BY LoanPurpose
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE LoanPurpose IS NULL;

-- ===========================================
-- STEP 4: Normalize boolean text fields to 1/0
-- ===========================================
UPDATE loan_data SET HasMortgage = CASE WHEN LOWER(HasMortgage) = 'yes' THEN 1 ELSE 0 END;
UPDATE loan_data SET HasDependents = CASE WHEN LOWER(HasDependents) = 'yes' THEN 1 ELSE 0 END;
UPDATE loan_data SET HasCoSigner = CASE WHEN LOWER(HasCoSigner) = 'yes' THEN 1 ELSE 0 END;

-- ===========================================
-- STEP 5: Correct common typos in categorical fields
-- ===========================================

UPDATE loan_data SET Education = 'Bachelor''s'
WHERE LOWER(Education) LIKE '%chelor%' OR LOWER(Education) LIKE 'bach%lor%' OR LOWER(Education) LIKE 'bachalor%';

UPDATE loan_data SET Education = 'Master''s'
WHERE LOWER(Education) LIKE 'm%ster%' OR LOWER(Education) LIKE 'masster%' OR LOWER(Education) LIKE 'mastor%';

UPDATE loan_data SET Education = 'High School'
WHERE LOWER(Education) LIKE 'h%gh school%' OR LOWER(Education) LIKE 'high scool%' OR LOWER(Education) LIKE 'hig school%';

-- ===========================================
-- STEP 6: Remove extreme numeric outliers
-- ===========================================
DELETE FROM loan_data
WHERE CreditScore < 300 OR CreditScore > 850;

-- ===========================================
-- STEP 7: Deduplicate by LoanID (requires unique ID column)
-- ===========================================
DELETE FROM loan_data
WHERE id NOT IN (
    SELECT MIN(id)
    FROM loan_data
    GROUP BY LoanID
);
