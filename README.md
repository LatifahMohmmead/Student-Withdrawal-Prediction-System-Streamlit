What is this project?

This project is a case study for Data Science Academy, an online school that noticed many students fall behind or withdraw before finishing their courses. Advisors usually find out too late — after a missed assignment or a failed exam — when it's hard to help anymore.

The goal is to use data the academy already collects (enrollment info, demographics, assessment scores, and website activity logs) to flag at-risk students early, and explain why a student was flagged, so advisors can offer the right kind of support instead of a generic warning.

The data used is the public OULAD dataset (Open University Learning Analytics Dataset).
What does the Streamlit app do?

The app is a simple, visual tool an advisor could use, with no coding needed:

You pick a student's details using sliders and dropdowns in the sidebar:
Mean assessment score
Total clicks on the course website (engagement)
Registration date (in days, relative to course start)
Socio-economic band (IMD band)
Highest education level
Click "Predict now". The app runs a Random Forest model trained on real student data and tells you:
Whether this student is likely to withdraw or not
The exact probability (as a percentage)
You get a chart showing the probability split (withdraw vs. continue), plus a second chart showing which factors mattered most to the model overall — so the result isn't just a black-box guess.
