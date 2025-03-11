# kawin

Python implementation of the Kampmann-Wagner Numerical (KWN) model to predict precipitate nucleation and growth behavior. This package couples with pycalphad to perform thermodynamic and kinetic calculations.

Notes
-----
There has been a lot of changes in the API between the current release and the development version of kawin. The current examples reflect the API of the development version. Examples using the release API can be found in commit f748761. Users of the development version are encouraged to switch to the new API. While back-compatibility was sought to be maintained, there may have been some changes. If there are any issues in running the development or release version of kawin, please open an issue.

Installation
------------
`pip install kawin`

Examples
--------
Examples on Jupyter notebooks can be found on [NBViewer](https://nbviewer.org/github/materialsgenomefoundation/kawin/tree/main/examples/).

Dependencies
------------
numpy, scipy, matplotlib, pycalphad
