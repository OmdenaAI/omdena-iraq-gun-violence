# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:36:27 2021

@author: shyam
"""

import twint
import nest_asyncio
import pandas as pd

nest_asyncio.apply()


class TwitterScraper():
    
    def __init__(self, twint_columns, columns, limit, language="ar", 
                 enable_pandas=True, hide_output=True): 
     
        self.twint_columns = twint_columns
        self.columns = columns
        
        self.twitter = twint.Config()
        self.twitter.Limit = limit
        self.twitter.Lang = language
        self.twitter.Pandas = enable_pandas
        self.twitter.Hide_output = hide_output
        
    
    def scrape(self, keyword):
        
        self.twitter.Search = keyword
        twint.run.Search(self.twitter)
        self.final_df = twint.storage.panda.Tweets_df
        if self.final_df.shape[0]!=0: 
            self.final_df = self.final_df[self.twint_columns]
            self.final_df.columns = self.columns[:-1]
            self.final_df.loc[:, self.columns[-1]] = "Twitter" 
        else:
            self.final_df = pd.DataFrame(self.columns)
        
        return self.final_df
