DROP TABLE IF EXISTS loans;
CREATE TABLE loans (
    loan_id INT PRIMARY KEY,
    region VARCHAR(50),
    loan_type VARCHAR(50),
    approved_amount DECIMAL(10,2),
    repayment_period INT,
    delayed_payments INT,
    applicant_age INT,
    applicant_gender VARCHAR(10)
);

INSERT INTO loans VALUES
(1, 'North', 'Home', 250000, 120, 3, 35, 'Male'),
(2, 'South', 'Auto', 15000, 36, 0, 29, 'Female'),
(3, 'East', 'Personal', 5000, 24, 1, 42, 'Male'),
(4, 'West', 'Home', 300000, 180, 5, 50, 'Female');
