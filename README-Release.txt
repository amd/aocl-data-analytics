AOCL-DA is a data analytics library providing optimized building blocks for data analysis. It is written with a C-compatible interface to make it as seemless as possible to integrate with the library from whichever programming language you are using. The intended workflow for using the library is as follows:
-- load data from memory by reading CSV files or using the in-built da_datastore object
-- preprocess the data by removing missing values, standardizing, and selecting certain subsets of the data, before extracting contiguous arrays of data from the da_datastore objects
-- data processing (e.g. principal component analysis or linear model fitting)

C++ example programs (instructions for their compilation) can be found in the examples folder of your installation.

For further details on the library contents, please refer to the user guide.

AOCL-DA is developed and maintained by AMD. For support or queries, you can email us on toolchainsupport@amd.com.