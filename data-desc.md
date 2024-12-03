# GDELT

- date (datetime) : The date of the event in YYYYMMDD format


- Actor1CountryCode (string) : 3-character CAMEO code for country affiliation of Actor1, blank if system was unable to define it  

- Actor1Type1Code (string) : 3-character CAMEO code of the CAMEO “type” or “role” of Actor1

- Actor1Type (string) : [ GOV, Opposition, Business, Benevolent, Undefined ]

- Actor1Geo_CountryCode (string) : 2-character FIPS10-4 country code for the location


- Actor2CountryCode (string) : ... for Actor2.  

- Actor2Type1Code (string) : ... for Actor2.

- Actor2Type (string) : ... for Actor2.

- Actor2Geo_CountryCode (string) : ... for Actor2.  


- Importance (num) : 0..25000, uses number of mentions, articles, sources and avgTone

- Impact (num) : -1e6..1e6

- ImpactBin (string) : 
	[          'VeryNeg',   'Neg',    'SlightNeg', 'Neutral', 'SlightPos',  'Pos',    'VeryPos'       ]
	[ -10000000,      -20000,    -5000,         -10,        10,          5000,   20000,       10000000]


- EventCode (string) : 3..4 char. This is the raw CAMEO action code describing the action that Actor1 performed upon Actor2

- EventType (string) : type of event in words, e.g. 'Statement', 'Appeal', 'Aid', ...

- QuadClass (int) : [ VerbalCoop, MaterialCoop, VerbalConf, MaterialConf ]

- GoldsteinScale (float) : -10 to +10, theoretical potential impact that type of event will have on the stability of a country

- NumMentions (int) : it is recommended that this field be normalized by the average or other measure of the universe of events during the time period of interest

- NumSources (int) : it is recommended that this field be normalized by the average or other measure of the universe of events during the time period of interest

- NumArticles (int) : it is recommended that this field be normalized by the average or other measure of the universe of events during the time period of interest

- AvgTone (numeric) : average “tone”, -100..+100. Common values -10..+10, with 0 indicating neutral. This can be used as a method of filtering the “context” of events as a subtle measure of the importance of an event and as a proxy for the “impact” of that event. For example, a riot event with a slightly negative average tone is likely to have been a minor occurrence, whereas if it had an extremely negative average tone, it suggests a far more serious occurrence. A riot with a positive score likely suggests a very minor  occurrence described in the context of a more positive narrative (such as a report of an attack occurring in a discussion of improving conditions on the ground in a country and how the number of attacks per day has been greatly reduced).  

Reference: [GDELT Event Codebook V2.0](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf)




# BTC
- **date** (*datetime*) : The date of the price data.
- **Open** (*float*) : Price at the start of the day.
- **High** (*float*) : Highest price during the day.
- **Low** (*float*) : Lowest price during the day.
- **Close** (*float*) : Price at the end of the day.
- **Volume** (*float*) : Amount of Bitcoin traded during the day.
- **Volume$** (*float*) : Total volume in USD traded during the day.
- **Change** (*float*) : Change in price from Open to Close.
- **Change%** (*float*) : Percentage change in price from Open to Close.
- **Fluctuation** (*float*) : Difference between High and Low prices.
- **Fluctuation%** (*float*) : Percentage fluctuation between High and Low prices.