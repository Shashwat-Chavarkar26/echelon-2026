# Echelon-2026
The following project was made for a 24-hour hackathon in my college. 
The problem statement given to us was to make a sentiment and lifestyle tracker(autonomous agent). The domain given to us was to build a predictive analytics and visualization tool that forecasts **silver price trends in INR** using **historical data** and **sentiment analysis**.
## Overview
The **Silver Prediction Model** is designed to analyze and visualize the trends of silver prices over time.  
It uses live market data from **Yahoo Finance** (`yfinance` API) and applies custom **sentiment labeling** to identify market phases such as:
-**Birth** → Stable/Neutral  
- **Growth** → Rising trend  
- **Peak** → High momentum  
- **Reversal** → Downturn starts  
- **Death** → Sharp decline

