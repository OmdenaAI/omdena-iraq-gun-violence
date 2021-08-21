# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 02:06:51 2021

@author: shyam
"""

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class SMScraperOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SMSrcaper options")

        # Scrapper choices
        self.parser.add_argument("--choices", 
                                 nargs="+", 
                                 type=int, 
                                 help="scraper choices - {0: Twitter, 1: Reddit}", 
                                 default=[0, 1])
        
        
        # Common options
        self.parser.add_argument("--columns",
                                 type=str,
                                 help="columns to choose", 
                                 nargs="+",
                                 default=["text", "tags", "username", "link", "source"]) 
        self.parser.add_argument("--ar_keywords_path", 
                                 type=str,
                                 help="path for keywords in Arabic",
                                 default=os.path.join(file_dir, "inputs/violent_words.txt"))
        self.parser.add_argument("--output_data_path", 
                                 type=str,
                                 help="path for data created",
                                 default=os.path.join(file_dir, "data/violence_data.csv"))
        self.parser.add_argument("--log",
                                 type=str,
                                 help="path to log file", 
                                 default=os.path.join(file_dir, "log/out_log.txt"))
        
        
         # Twitter options
        self.parser.add_argument("--twint_columns",
                                 type=str,
                                 help="required_columns for twint capturing \
                                     should be equivalent to columns except source",
                                 default=["tweet", "hashtags", "username", "link"])
        self.parser.add_argument("--capture_limit",
                                 type=int,
                                 help="approximate number of tweets to be captured per keyword",
                                 default=1000)
        
        
         # Reddit options
        self.parser.add_argument("--reddit_config_path",
                                 type=str,
                                 help="path to the reddit_config file for PRAW",
                                 default=os.path.join(file_dir, "inputs/reddit_config.json"))
        self.parser.add_argument("--praw_params",
                                 type=str,
                                 help="Search paramters for praw",
                                 default={'sort':'relevance', 'limit':None, 'syntax':'cloudsearch'})
        
    
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
