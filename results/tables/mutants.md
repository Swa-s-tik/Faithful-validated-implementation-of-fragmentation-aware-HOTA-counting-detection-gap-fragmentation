| Mutation | Short sequences, cases caught of 5700 | Long sequences with ties, cases caught of 1900 |
|---|---|---|
| PoseTrack21 rule (break only on identity change) | 3596 | 1863 |
| drop the predicted-identity break (step 4, second test) | 1581 | 1331 |
| drop the ground-truth break (step 4, first test) | 1634 | 1384 |
| window includes the earlier match frame | 3232 | 1647 |
| window skips the first frame after the earlier match | 2825 | 1554 |
| strict threshold cut (s > alpha) | 0 | 1273 |
| sort by frame before identities | 2024 | 910 |
| denominator without the overlap term | 5320 | 1898 |
