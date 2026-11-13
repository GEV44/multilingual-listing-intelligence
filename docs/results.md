### Extraction (rules extractor, field-level)

| field | precision | recall | F1 | support |
|---|---|---|---|---|
| deal_type | 0.999 | 0.959 | 0.979 | 1000 |
| price | 0.991 | 0.767 | 0.865 | 1000 |
| currency | 1.000 | 0.774 | 0.873 | 1000 |
| rooms | 1.000 | 0.771 | 0.870 | 955 |
| area_m2 | 1.000 | 0.749 | 0.856 | 900 |
| floor | 0.917 | 0.784 | 0.846 | 849 |
| floors_total | 1.000 | 0.707 | 0.828 | 727 |
| district | 1.000 | 0.937 | 0.967 | 872 |
| street | 1.000 | 1.000 | 1.000 | 547 |
| renovation | 1.000 | 0.759 | 0.863 | 636 |
| furniture | 1.000 | 0.712 | 0.832 | 500 |
| pets | 1.000 | 0.768 | 0.869 | 319 |
| heating | 1.000 | 0.750 | 0.857 | 408 |
| **micro-avg** | 0.991 | 0.808 | 0.890 | 9713 |

| slice | micro P | micro R | micro F1 | macro F1 |
|---|---|---|---|---|
| lang:en | 1.000 | 0.831 | 0.908 | 0.904 |
| lang:hy | 0.988 | 0.805 | 0.887 | 0.883 |
| lang:ru | 0.988 | 0.793 | 0.880 | 0.872 |
| tier:hard | 0.978 | 0.620 | 0.759 | 0.738 |
| tier:standard | 1.000 | 0.986 | 0.993 | 0.994 |

### Ranking (100 held-out synthetic requests)

| system | nDCG@5 | nDCG@10 |
|---|---|---|
| random (after hard filters) | 0.501 | 0.533 |
| price only (after hard filters) | 0.417 | 0.480 |
| hand-set weights | 0.746 | 0.723 |
| learned weights | 0.798 | 0.765 |
| hand-set weights, gold facts (upper bound) | 0.946 | 0.927 |

### De-duplication (394 posts of 204 flats, pairwise)

| method | precision | recall | F1 |
|---|---|---|---|
| photo pHash only | 1.000 | 0.580 | 0.734 |
| spec match only | 0.774 | 0.988 | 0.868 |
| combined (default) | 0.874 | 0.972 | 0.920 |
