Deep Learning Challenge: Charity Funding Predictor

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With the use of my knowledge of machine learning and neural networks, I used the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively


### Step 1: Preprocess the Data

I started by using Pandas and scikit-learn’s `StandardScaler()`, to preprocess the dataset. This step prepared me to compile, train, and evaluate the neural network model later in step 2.

1. Read in the charity_data.csv to a Pandas DataFrame, to identify the following in the dataset:
  * What variable(s) are the target(s) for the model?
  * What variable(s) are the feature(s) for the model?

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

### Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, I designed a neural network, and deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Then I compiled, train, and evaluated tje binary classification model to calculate the model’s loss and accuracy.

1. Continue using the Jupyter Notebook in which I performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. Added a second hidden layer with an appropriate activation function.

5. Created an output layer with an appropriate activation function.

6. Checked the structure of the model.

7. Compiled and train the model.

8. Created a callback that saves the model's weights every five epochs.

9. Evaluated the model using the test data to determine the loss and accuracy.

10. Saved and exported the results to an HDF5 file. Named  `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Using TensorFlow, I optimize the model to achieve a target predictive accuracy higher than 75%.

Using any or all of the following methods to optimize my model:

* Adjusted the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
* Added more neurons to a hidden layer.
* Added more hidden layers.
* Used different activation functions for the hidden layers.
* Added or reduced the number of epochs to the training regimen.

1. Created a new Jupyter Notebook file and named it `AlphabetSoupCharity_Optimzation.ipynb`.

2. Imported my dependencies and read in the `charity_data.csv` to a Pandas DataFrame.

3. Preprocessed the dataset like I did in Step 1, and adjusted for any modifications that came out of optimizing the model.

4. Designed a neural network model, and adjusted for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Saved and exported my results to an HDF5 file. Named `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model

Write a report on the performance of the deep learning model I created for AlphabetSoup.

The report contained the following:

1. **Overview** of the analysis: Explaining the purpose of this analysis.

2. **Results**: Using bulleted lists and images that support my answers, and address the following questions.

  * Data Preprocessing
    * What variable(s) are the target(s) for the model?
    * What variable(s) are the features for the model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?
  
* Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did were select for my neural network model, and why?
    * Was I able to achieve the target model performance?
    * What steps were taken in my attempts to increase model performance?

3. **Summary**: Summarize the overall results of the deep learning model. 
- - -



- - - 

© 2022 Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.	

