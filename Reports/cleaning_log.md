use USD as currency!

### Deleted Rows in clean_features
URL
Website
Tagline
Long description
Address
Street
Street Number
Street and street number
zipcode
HQ State
HQ City
Latitude
Longitude
World Locations
Custom HQ regions
Total funding (EUR M)
Closing year -> no data
Closing month -> no data
Closing date -> no data
KPI Currency
Valuation
Valuation Currency
Valuation (EUR)
Historical Valuations - values (EUR M)
AngelList
Facebook
Twitter
LinkedIn
Crunchbase
Google Play Link
iTunes
Each round Currency
Logo
Founders LinkedIn
App downliads ....
Website traffic rank
Employee rank
Number of alumni founders that raised > 10M
TechStack
Trade Register number, name, url
PIC Number
Dealroom Signal (all columns)
innovation corporate rank
Founding location (not enough data)
Team Dealroom (not enough data)
Lead Investors (not enough data)
Seed year (not enough data)
launch month (not enough data)
founders strength (not enough data)
Technologies (not enough data)
year company became unicorn (not enough data)
year company became future unicorn (not enough data)
Website traffic estimate (not relevant)
employees in hq country (not relevant)
revenue EUR (we use usd)
R&D margin (not enough data)
EV/Revenue (not enough data)
EV/EBITDA (not enough data)
website traffic estimate 6 months (not relevant)
public lists (text)
all tags (text)
founders statuses (after y)
Founders company experience
founders first degree
founders first degree year
is top past founder (not enough data)
Team (editorial)
each investor type (not relevant)
Income streams (not relevant)
number of patens (too many 0 values)
launch date (year is sufficient)
Tags (too much)
Employee growth (only last 3,6,12 months)







Original Target Variable:
(IPO) or (M&A) or ((survived 5+ years) and (20% or more revenue growth over 3 years)).

Change to Target Variable:
Financial data such as profit is highly distorted due to Corona pandemic that lead to negative margins. Hence, I prefer employee number over revenue: the values are not as volatile and profitable startups could still see growth during the pandemic.
=>
(IPO) or (M&A) or ((survived 5+ years) and (>20% employee CAGR over year 2 to year 5 after founding)).

for X:
remove all entries where Launch year < 2016
remove all entries where Launch year > 2021 (survival of 5+ years)

for y:
employee growth calculation
IPO/ACQUISITION in Last round column


ACQUISITION: identified problematic case in data
Last round: 
- ACQUISITION;BANKRUPTCY; ... or BUYOUT;BANKRUPTCY; .... no success, acquisition by lenders

Tried... 
(IPO) or (M&A) or ((survived 5+ years) and (>20% employee CAGR over year 2 to year 5 after founding)) target variable and got... 
IS_SUCCESS
0    0.94984
1    0.05016
... distribution with too little success rate. I also tried to lower the CAGR rules but without better results.

Then, I tried...
(IPO) or (M&A) or ((survived 5+ years) and (>20% employee CAGR over year 2 to year 5 after founding)) or (Scale-up (>50 employees as most recent value) as a proxy for long-term survival)
... and found a discrepancy in very high scale-ups (~1400) and only few CAGR (~200). I considered the low data density in Employees (2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026) column to be the cause.

So I tried the following: I calculated the dynamic CAGR over all years between the first and the last value I can find. The threshold is still 20% CAGR and >10 employees at the start. This is much more reliable, finds startups that cannot reach the >50 employee threshold but are still significantly growing. This gave me the following distribution of successful startups:

Breakdown:
High Growth (dynamic CAGR):    1040
Successful Exits:      628
Scale-ups (>50 Emp):   1488
TOTAL SUCCESSES (1):   2288
------------------------------
Class Distribution:
IS_SUCCESS
0    0.817252
1    0.182748
The metrics have a 72,25% overlap (convergent validity)


