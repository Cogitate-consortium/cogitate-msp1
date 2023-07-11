This script helps download the sample dataset from XNAT.

XNAT Credentials are required to download the data and the data will be downloaded in the raw XNAT data format. It is provided for demonstration / testing purpose only.

In order to access the XNAT instance a config file with
credentials have to be created in the below format.

Filename: .xnat_curate_netrc
```
machine xnat-curate.ae.mpg.de
login <username>
password <password>
```

Usage
1. Install xnat client using requirments_xnat.txt
2. Setup cofnig file as described above.
3. Modify script and run it accordingly.
