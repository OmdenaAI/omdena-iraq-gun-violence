# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 02:04:41 2021

@author: shyam
"""

import pandas as pd

import scrapers
from options import SMScraperOptions

from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")


class SMScraper():
    
    def __init__(self, options):
        
        self.opt = options
        self.log = []

        self.ar_keywords = self.get_keywords(self.opt.ar_keywords_path)
        
        self.scrappers_dict = {"Twitter" : scrapers.TwitterScraper,
                               "Reddit" : scrapers.RedditScraper}
        
        self.datasets = OrderedDict([(0, "Twitter"), (1, "Reddit")])
        self.scrapers = []
           
        for choice in self.opt.choices:
           name = self.datasets[choice]
           if choice == 0:
               scraper = self.scrappers_dict[name](twint_columns=self.opt.twint_columns,
                                                   columns=self.opt.columns,
                                                   limit=self.opt.capture_limit)  
           if choice == 1:
               scraper = self.scrappers_dict[name](config_file=self.opt.reddit_config_path,
                                                   columns=self.opt.columns,
                                                   search_params=self.opt.praw_params)
               
           self.scrapers.append(scraper)
           self.output_df = pd.DataFrame(columns=self.opt.columns)
        
        
    def get_keywords(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            keywords = f.readlines()
            
        return keywords
    
    
    def save_df(self, df):
        df.to_csv(self.opt.output_data_path, index=False)
        print("Output data file saved in {} path with {} entries"
              .format(self.opt.output_data_path, df.shape[0]))

        
    def create(self):
        for idx, word in enumerate(self.ar_keywords):
            print("Creating data for {} --> {} ".format(idx, word))
            for scraper in self.scrapers:
                final_df = scraper.scrape(word)
                self.__create_log(word, final_df.shape[0])
                self.output_df = self.output_df.append(final_df)
                
        self.output_df = self.output_df.drop_duplicates(subset=['text'], keep='first')
        self.save_df(self.output_df)
        self.__write_log(self.log)
        
                 
    def __create_log(self, word, size):
        self.log.append("Created {} entries for {} ".format(size, word))
        
    
    def __write_log(self, log):
        out_file = open(self.opt.log, "w+", encoding="utf-8")
        for element in self.log:
            out_file.write(element + "\n")
        print("Output log file saved in {} path"
              .format(self.opt.log))


if __name__ == "__main__":
    options = SMScraperOptions()
    opts = options.parse()
    TextScraper = SMScraper(opts)
    TextScraper.create()
    
    