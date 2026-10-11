"""Fixed vocabularies and encodings for the feature store.

Vocabularies are hard-coded, not learned, so encodings never depend on the training set.
"""

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur", "Journalist",
    "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]

LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
    "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan",
]

CREDIT_MIX_ORDINAL = {"Bad": 0, "Standard": 1, "Good": 2}
MIN_PAYMENT_BINARY = {"Yes": 1, "No": 0}
SPEND_LEVEL_ORDINAL = {"Low": 0, "High": 1}
PAYMENT_VALUE_ORDINAL = {"Small": 0, "Medium": 1, "Large": 2}

CLICKSTREAM_COLUMNS = [f"fe_{i}" for i in range(1, 21)]

# Age group bin names (one-hot + missing flag), matching the order in build_gold.py
AGE_BIN_NAMES = ["age_lt_25", "age_25_39", "age_40_54", "age_55_plus", "age_missing"]

# silver column -> gold feature name
NUMERIC_FEATURES = {
    "Age": "age",
    "Annual_Income": "annual_income",
    "Monthly_Inhand_Salary": "monthly_inhand_salary",
    "Num_Bank_Accounts": "num_bank_accounts",
    "Num_Credit_Card": "num_credit_card",
    "Interest_Rate": "interest_rate",
    "Num_of_Loan": "num_of_loan",
    "Delay_from_due_date": "delay_from_due_date",
    "Num_of_Delayed_Payment": "num_of_delayed_payment",
    "Changed_Credit_Limit": "changed_credit_limit",
    "Num_Credit_Inquiries": "num_credit_inquiries",
    "Outstanding_Debt": "outstanding_debt",
    "Credit_Utilization_Ratio": "credit_utilization_ratio",
    "Credit_History_Age_Months": "credit_history_age_months",
    "Total_EMI_per_month": "total_emi_per_month",
    "Amount_invested_monthly": "amount_invested_monthly",
    "Monthly_Balance": "monthly_balance",
}


def occupation_feature(occupation):
    return f"occupation_{occupation.lower()}"


def loan_type_feature(loan_type):
    """'Credit-Builder Loan' -> 'loan_type_credit_builder'."""
    return "loan_type_" + loan_type.lower().replace(" loan", "").replace("-", "_").replace(" ", "_")
