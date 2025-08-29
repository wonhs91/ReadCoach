
MOCK BOOK RECOMMENDATION DATASET
================================

Files:
- books.csv               (n=6428 rows)
- users.csv               (n=1500 rows)
- interactions.csv        (n=82392 rows)
- interactions_train.csv  (time-aware split: training)
- interactions_val.csv    (time-aware split: validation)
- interactions_test.csv   (time-aware split: test)

Schema:
books.csv
---------
book_id         string  Edition-level id (e.g., 'b_000123')
work_id         string  Canonical work id; multiple editions map to the same work
title           string
author_id       string
author_name     string
genre           string
pub_year        int
series          string  Empty if standalone
series_index    int     0 if standalone
format          string  'hardcover' | 'paperback' | 'ebook'

users.csv
---------
user_id         string
age             int
country         string (ISO-like)
preferred_genres string  semicolon-separated list
favorite_authors string  semicolon-separated list of author_ids

interactions*.csv
-----------------
user_id         string
book_id         string (edition level)
work_id         string (canonical; useful if you want to aggregate editions)
event           string ('view','wishlist','purchase')
weight          int    implicit weight mapping: view=1, wishlist=2, purchase=5
rating          float  1..5 only present (mostly) for purchases; NaN otherwise
ts              datetime  ISO timestamp

Notes:
- Popularity is heavy-tailed (Zipfian). Users have genre & author affinities.
- Younger users are biased toward newer pub_years.
- Ratings correlate with 'true_quality' per work and user taste, plus noise.
- For pure implicit CF: use 'weight' and ignore 'rating'.
- For edition vs. work modeling: aggregate book_id -> work_id if desired.

Suggested usage:
- Build a user–item matrix from interactions_train.csv
- Evaluate Recall@K on the held-out interactions_val/test.csv
- Item–Item kNN works best on work_id (aggregate editions).

Reproducibility:
- Generated with fixed RNG seed (42).
