# -*- coding: UTF-8 -*-
# Amazon Dataset Processing

import sys
sys.path.append("../..")

import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
from utils import Logger

class AmazonDatasetProcessor():
    def __init__(self):
        self.amazon_reviews_path = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
        self.amazon_meta_path    = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"
        self.logger              = Logger(file="./amazon.log")
    
    def _download(self):
        self.logger.log("INFO", "Begin to download Amazon Dataset to './raw_data'.")
        if os.path.exists("./raw_data"):
            os.system("rm -r ./raw_data/*")
        else:
          os.system("mkdir ./raw_data")
        os.system("cd ./raw_data")
        # download review data
        self.logger.log("INFO", "Downloading reviews data ...")
        os.system("wget -c {}".format(self.amazon_reviews_path))
        os.system("gzip -d reviews_Electronics_5.json.gz")
        # download meta data
        self.logger.log("INFO", "Downloading meta data ...")
        os.system("wget -c {}".format(self.amazon_meta_path))
        os.system("gzip -d meta_Electronics.json.gz")
        self.logger.log("INFO", "Finish download!")
        os.system("cd ../")

    def _read_to_df(self, path:str) -> pd.DataFrame:
        assert os.path.exists(path), "'{}' does not exist!".format(path)
        df = {}
        with open(path, 'r') as f:
            i = 0
            for line in f:
                df[i] = eval(line)
                i += 1
            f.close()
        df = pd.DataFrame.from_dict(df, orient='index')
        return df
    
    def _build_map(df:pd.DataFrame, col_name:str) -> tuple:
        key          = sorted(df[col_name].unique().tolist())
        m            = dict(zip(key, range(len(key))))
        df[col_name] = df[col_name].map(lambda x: m[x])
        return m, key

    def _convert_to_pkl(self):
        assert os.path.exists("./raw_data"), "'./raw_data' does not exist!"
        assert os.path.exists("./raw_data/reviews_Electronics_5.json"), \
            "'./raw_data/reviews_Electronics_5.json' does not exist!"
        assert os.path.exists("./raw_data/reviews_Electronics_5.json"), \
            "'./raw_data/meta_Electronics.json' does not exist!"
        assert not os.path.exists("./raw_data/remaped_amazon_electronics.pkl"), \
            "'./raw_data/remaped_amazon_electronics.pkl' already exists!"

        review_df = self._read_to_df("./raw_datareviews_Electronics_5.json")
        meta_df   = self._read_to_df("./raw_datameta_Electronics.json")
        meta_df   = meta_df[meta_df['asin'].isin(review_df['asin'].unique())]
        meta_df   = meta_df.reset_index(drop=True)

        reviews_df            = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
        meta_df               = meta_df[['asin', 'categories']]
        meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

        asin_map, asin_key = self._build_map(meta_df, 'asin')
        cate_map, cate_key = self._build_map(meta_df, 'categories')
        revi_map, revi_key = self._build_map(reviews_df, 'reviewerID')

        user_count, item_count, cate_count, example_count = \
            len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]

        meta_df            = meta_df.sort_values('asin')
        meta_df            = meta_df.reset_index(drop=True)
        reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
        reviews_df         = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
        reviews_df         = reviews_df.reset_index(drop=True)
        reviews_df         = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
        cate_list          = [meta_df['categories'][i] for i in range(len(asin_map))]
        cate_list          = np.array(cate_list, dtype=np.int32)

        with open('./raw_data/remaped_amazon_electronics.pkl', 'wb') as f:
            pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
            pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
            pickle.dump((user_count, item_count, cate_count, example_count),f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
            f.close()

    def _build_datasets(self):
        assert not os.path.exists("./raw_data/amazon_electronics_datasets.pkl"), \
            "'amazon_electronics_datasets.pkl' already exists!"
        self.logger.log("INFO", "Building amazon electronics datasets ...")
        
        with open('./raw_data/remaped_amazon_electronics.pkl', 'rb') as f:
            reviews_df = pickle.load(f)
            cate_list = pickle.load(f)
            user_count, item_count, cate_count, example_count = pickle.load(f)
            f.close()

        train_set = []
        test_set = []
        for reviewerID, hist in reviews_df.groupby('reviewerID'):
            pos_list = hist['asin'].tolist()
            def gen_neg():
                neg = pos_list[0]
                while neg in pos_list:
                    neg = random.randint(0, item_count-1)
                return neg
            neg_list = [gen_neg() for i in range(len(pos_list))]

            for i in range(1, len(pos_list)):
                hist = pos_list[:i]
                if i != len(pos_list) - 1:
                    train_set.append((reviewerID, hist, pos_list[i], 1))
                    train_set.append((reviewerID, hist, neg_list[i], 0))
                else:
                    label = (pos_list[i], neg_list[i])
                    test_set.append((reviewerID, hist, label))

        random.shuffle(train_set)
        random.shuffle(test_set)

        train_size = len(train_set)
        test_size  = len(test_set)

        assert len(test_set) == user_count

        with open('./raw_data/amazon_electronics_datasets.pkl', 'wb') as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
            f.close()

        self.logger.log("INFO", "---- Dataset Details -----")
        self.logger.log("INFO", "Train size   : ", train_size)
        self.logger.log("INFO", "Test size    : ", test_size)
        self.logger.log("INFO", "User count   : ", user_count)
        self.logger.log("INFO", "Item count   : ", item_count)
        self.logger.log("INFO", "Cate count   : ", cate_count)
        self.logger.log("INFO", "Example count: ", example_count)
        self.logger.log("INFO", "--------------------------")
        self.logger.log("INFO", "Save dataset to './raw_data/amazon_electronics_datasets.pkl'.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action="store_true", default=False)
    parser.add_argument('--random_seed', type=int, default=1234)
    return parser.parse_args()

if __name__ == "__main__":
    # parse args
    args = get_args()
    # set random seed
    random.seed(args.random_seed)
    processor = AmazonDatasetProcessor()
    if args.download:
        processor._download()
    processor._convert_to_pkl()
    processor._build_datasets()
    