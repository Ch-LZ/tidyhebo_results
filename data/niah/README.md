# NIAH oracle assets

`python run_experiments.py setup niah` downloads the two source forests from
PV-Lab/ZoMBI commit `04a7f0fa1a1a91c2afe60f1b0c60138e21c0de99`, verifies their
SHA-256 digests and exports portable NumPy trees in a separate conversion
environment. The runtime does not load sklearn pickles.

The large generated `.npz` forests and downloaded source pickles are local
installation assets and are excluded from the source archive. Their metadata,
content identities, converter, source digests and upstream MIT license are
retained here. The generated content identities were checked against the
identities recorded in the paper campaign.

Historical optimization curves used to draw the paper are separate, tracked
data under `data/historical/`; plotting does not need these trained forests.
