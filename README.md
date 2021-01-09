[image1]: assets/hist_box.png "image1"
[image2]: assets/shape.png "image2"
[image3]: assets/stat_prop.png "image3"
[image4]: assets/binom_dis.png "image4"
[image5]: assets/bayes.png "image5"
[image6]: assets/not_1.png "image6"
[image7]: assets/not_2.png "image7"
[image8]: assets/sampling_dis.png "image8"
[image9]: assets/sampling_dis_2.png "image9"
[image10]: assets/bootstrap.png "image10"
[image11]: assets/confidence.png "image11"
[image12]: assets/confidence_1.png "image12"
[image13]: assets/confidence_2.png "image13"
[image14]: assets/hypo_guide.png "image14"
[image16]: assets/norm.png "image16"
[image17]: assets/norm_eq.png "image17"
[image18]: assets/std_norm_eq.png "image18"
[image19]: assets/h0_h1.png "image19"
[image20]: assets/norm_oneside_twoside.png "image20"
[image21]: assets/anova.png "image21"
[image22]: assets/accept_reject.png "image22"
[image23]: assets/one_sided.png "image23"
[image24]: assets/two_sided.png "image24"
[image25]: assets/stat_error.png "image25"
[image26]: assets/stat_error2.png "image26"
[image27]: assets/p_value_3.png "image27"
[image28]: assets/hypo_test_table.png "image28"
[image29]: assets/critical_val.png "image29"
[image30]: assets/type_1.png "image30"
[image31]: assets/type_2.png "image31"
[image32]: assets/types_of_error.png "image32"
[image33]: assets/p_value_1.png "image33"
[image34]: assets/p_value_2.png "image34"
[image35]: assets/hypo_conclusion.png "image35"
[image36]: assets/large_sample_size.png "image36"
[image37]: assets/confidence_hypo.png "image37"
[image38]: assets/hypo_summary.png "image37"

# Practical Statistics 

## Outline
- [Descriptive vs Inferential Statistics](#descr_infer)
- [Descriptive Statistics](#dscr)
- [Probability](#Probability)
- [Inferential Statistics](#infer)


- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Descriptive vs Inferential Statistics  <a name="descr_infer"></a>
- ***Descriptive statistics*** is about describing collected data
- ***Inferential Statistics*** is about using collected data to draw conclusions to a larger population


    - ***Population*** - entire group of interest
    - ***Parameter*** - numeric summary about a population
    - ***Sample*** - subset of the population
    - ***Statistic*** - numeric summary about a sample

    ![image6]

# Descriptive Statistics <a name="dscr"></a>
## What is data?
- Data can come in sin many forms: 
    - structured data (numerical data, relational databases)
    - unstructured data (images, audio, video)

## Data types
- ***Quantitative***: numeric values 
    - ***discrete***: Pages in a Book, Trees in Yard, Dogs at a Coffee Shop
    - ***continuous***: Height, Age, Income
- ***Categorical***: group or set of items
    - ***ordinal***: Letter Grade, Survey Rating
    - ***nominal***: Gender, Marital Status, Breakfast Items

## Analyzing Quantitative Data
Four Aspects for Quantitative Data
- Measures of Center
- Measures of Spread
- The Shape of the data
- Outliers

## Analyzing Categorical data
by looking at the counts or proportion of individuals that fall into each group.

## Measures of Center
- ***Mean*** - average or the expected value

    <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{n} \sum_{i=1}^{n}  x_{i}" width="120px">
    
    ```
    # return the mean 
    df.mean(axis=None, skipna=None, level=None, numeric_only=None, **kwargs) 
    ```
- ***Median*** - value where 50% of the data are smaller and 50% are larger
    ```
    # return the median 
    df.median(axis=None, skipna=None, level=None, numeric_only=None, **kwargs) 
    ```
- ***Mode*** - the most frequently observed value in the dataset
    ```
    # return the mode 
    df.mode(axis=0, numeric_only=False, dropna=True) 
    ```

Remember: 
- In order to compute the median we MUST sort our values first
- If two (or more) numbers share the maximum value, then there is more than one mode

## Notation
- universal language used by academic and industry professionals to convey mathematical ideas
- e.g. plus, minus, multiply, division, and equal signs
- ***Random variables*** are represented by capital letters, e.g. X, Y, Z
- ***Observations of the random variable*** are represented by lowercase letters, e.g. x1, x2, x3
- P(X > 20) = x
- Aggregations: for sum sign, product sign, integration, etc.

## Measurs of Spread
- Use histograms for visualization data in certain bins (e.g. 1-4, 5-8, 9-12, 13-16)
- Use boxplot for analyse median and quartiles, maximum, minimum, IQR
1. Use the ***5 number summary***

    - ***Minimum***: The smallest number in the dataset.
    - ***Q1***: The value such that 25% of the data fall below.
    - ***Q2***: The value such that 50% of the data fall below.
    - ***Q3***: The value such that 75% of the data fall below.
    - ***Maximum***: The largest value in the dataset.

    In Addition:
    - ***Range***: difference between the maximum and the minimum
    - ***IQR***: The interquartile range is calculated as the difference between Q3 and Q1

    ![image1]

    ```
    # calculate min, max, quantiles
    df.min(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
    df.max(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
    df.quantile(q=0.5, axis=0, numeric_only=True, interpolation='linear')
    ```

2. Use ***Standard Deviation***
    - ***Variance (population)*** - squared representation of the average distance of each observation from the mean

        <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{n} \sum_{i=1}^{n}  (x_{i} - \bar{x})^2" width="200px">

    - ***Variance (sample)*** - squared representation of the average distance of each observation from the mean

        <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{n-1} \sum_{i=1}^{n-1}  (x_{i} - \bar{x})^2" width="200px">

    - ***Standard Deviation (population)*** - the average distance of each observation from the mean

        <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{1}{n} \sum_{i=1}^{n}  (x_{i} - \bar{x})^2}" width="200px">

    - ***Standard Deviation (sample)*** - the average distance of each observation from the mean

        <img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{1}{n-1} \sum_{i=1}^{n-1}  (x_{i} - \bar{x})^2}" width="200px">

    ```
    # calculate standard deviation
    df.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs)
    ```
## Shape
![image2]

| Shape     | Mean vs. Median     | Real World Applications
| :------------- | :------------- | :------------- |
| Symmetric (Normal) |	Mean equals Median |	Height, Weight, Errors, Precipitation
| Right-skewed 	| Mean greater than Median |	Amount of drug remaining in a blood stream, Time between phone calls at a call center, Time until light bulb dies
| Left-skewed 	| Mean less than Median | Grades as a percentage in many universities, Age of death, Asset price changes

```
# calculate skewness
df.skew(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
```

## Outliers
- Use e.g. histograms to detect outliers
- Use the Tukey rule

    - Find the first quartile Q1 (ie .25 quantile)
    - Find the third quartile Q3 (ie .75 quantile)
    - Calculate the inter-quartile range IQR (Q3 - Q1)
    - Any value that is greater than Q3 + 1.5 * IQR is an outlier
    - Any value that is less than Qe - 1.5 * IQR is an outlier

    ```
    def tukey_rule(data_frame, column_name):
  """ use Tukey rule to detect outliers in a dataframe column, output a data_frame with the outliers eliminated

     INPUT:
     ------------
     data_frame - DataFrame which has to be checked for outliers
     column_name - column to focus on for outlier check

     OUTPUT:
     ------------
     df_cleaned - DataFrame cleaned from outliers
    """

    # Calculate the first quartile of the population values
    Q1 = data_frame[column_name].quantile(0.25)

    # Calculate the third quartile of the population values
    Q3 = data_frame[column_name].quantile(0.75)

    # Calculate the interquartile range Q3 - Q1
    IQR = Q3 - Q1

    # Calculate the maximum value and minimum values according to the Tukey rule
    # max_value is Q3 + 1.5 * IQR while min_value is Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR

    # Filter the column_name data values that are greater than max_value or less than min_value
    df_cleaned = data_frame[(data_frame[column_name] < max_value) & (data_frame[column_name] > min_value)]

    return df_cleaned
    ```

# Probability <a name="Probability"></a>

![image3]

- with ***probability*** you predict data
- with ***statistics*** you use data to predict

A probability summary:
- P = Probility of an event 
- 1-P = probability of the opposite event
- P*P*P*P*P*... = Probaility of composite (independent) event

## Conditional Probability
We can formulate conditional probabilities for any two events in the following way:

The likelihood of event A occurring given that B is true:

- <img src="https://render.githubusercontent.com/render/math?math=P(A | B) = \frac{P(A \cap B)}{P(B)}" height="60px">

- <img src="https://render.githubusercontent.com/render/math?math=P(A) = P(A|B)P(B) %2B P(A|\bar{B})P(\bar{B})" height="40px">

## Bayes Rule

- <img src="https://render.githubusercontent.com/render/math?math=P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}" height="60px">

- <img src="https://render.githubusercontent.com/render/math?math=P(B) = P(B | A) \cdot P(A) %2B P(B | \bar{A}) \cdot P(\bar{A})" height="40px">

Remember:
- Calculate the prior probability 
- Calculate the normalizer
- Calculate the posterior probability

Example: C = Cancer, Pos = Test positive, Neg = Test negative

- P(C) = 0.01              
- P(Pos | C) = 0.9
- P(Neg | not C) = 0.9
- P(not C) = 0.99
- P(Neg | C) = 0.1
- P(Pos | not C) = 0.1

Test = Neg
Prior probability
- P(C | Neg) = P(C) * P(Neg | C) = 0.001
- P(not C | Neg) = P(not C) * P(Neg | not C) = 0.891

Normalizer
- P(Neg) =  P(C | Neg) + P(not C | Neg) = 0.892

Posterior probability
- P(C | Neg) = P(C) * P(Neg | C) / P(Neg) = 0.0011
- P(not C | Neg) = P(not C) * P(Neg | not C) / P(Neg) = 0.9989


![image5]


## Simulating Coins Flips in Python 
- Open notebook under ```notebooks/simulating_coin_flips.ipynb```
    ```
    import numpy as np
    import matplotlib.pyplot as plt
    % matplotlib inline

    # outcome of one coin flip
    np.random.randint(2)

    # outcomes of ten thousand coin flips
    np.random.randint(2, size=10000)

    # mean outcome of ten thousand coin flips
    np.random.randint(2, size=10000).mean()

    # outcome of one coin flip
    np.random.choice([0, 1])

    # outcome of ten thousand coin flips
    np.random.choice([0, 1], size=10000)

    # mean outcome of ten thousand coin flips
    np.random.choice([0, 1], size=10000).mean()

    # outcomes of ten thousand biased coin flips
    np.random.choice([0, 1], size=10000, p=[0.8, 0.2])

    # mean outcome of ten thousand biased coin flips
    np.random.choice([0, 1], size=10000, p=[0.8, 0.2]).mean()
    ```


# Binominal Distribution <a name="binom_dis"></a>

![image4]

where ***n*** is the number of events, ***k*** is the number of "successes", and ***p*** is the probability of "success".


## Simulating many Coins Flips in Python 
- Open notebook under ```notebooks/simulating_many_coin_flips.ipynb```
    ```
    import numpy as np

    # number of heads from 10 fair coin flips
    np.random.binomial(10, 0.5)

    # results from 20 tests with 10 coin flips
    np.random.binomial(10, 0.5, 20)

    # mean number of heads from the 20 tests
    np.random.binomial(10, 0.5, 20).mean()

    # reflects the fairness of the coin more closely as # tests increases
    np.random.binomial(10, 0.5, 1000000).mean()

    import matplotlib.pyplot as plt
    % matplotlib inline

    plt.hist(np.random.binomial(10, 0.5, 1000000));

    # gets more narrow as number of flips increase per test
    plt.hist(np.random.binomial(100, 0.5, 1000000));
    ```
# Interference Statistics <a name="infer"></a>

# [Sampling Distribution](https://www.slideshare.net/DonnaWiles1/sampling-distribution-84492010)
A sampling distribution is the distribution of a statistic.

![image8] ![image9]

Images are only valid in combination with Central Limit Theorem

Remember:
- The ***population distribution*** shows the values of the variable for all individuals in the population
- The ***distribution of sample data*** shows the values of the variable for all the individuals in the sample
- The ***sampling distribution*** shows the statistic values from all the possible samples of the same size from the population

- Open notebook under ```notebooks/Sampling Distributions-Solution.ipynb```

    ```
    sample_props = []
    for _ in range(10000):
        sample = np.random.choice(students, 5, replace=True)
        sample_props.append(sample.mean())

    # mean of sampling distribution
    sample_props = np.array(sample_props)
    sample_props.mean()
    ```
    

# Law of Large Numbers
The larger the sample size -> The closer the statistic gets to the parameter


# Central Limit Theorem
The Central Limit Theorem states that with a large enough sample size the sampling distribution of the mean will be normally distributed.

The Central Limit Theorem actually applies for these well known statistics:

- Sample means <img src="https://render.githubusercontent.com/render/math?math=\bar{x}" height="20px">
- Sample proportions <img src="https://render.githubusercontent.com/render/math?math=p" height="20px">
- Difference in sample means <img src="https://render.githubusercontent.com/render/math?math=\bar{x}_{1} - \bar{x}_{2}" height="25px">
- Difference in sample proportions <img src="https://render.githubusercontent.com/render/math?math=p_{1} - p_{2}" height="20px">

- Open notebook under ```notebooks/Sampling Distributions ... Central Limit Theorem.ipynb```

    ```
    import numpy as np
    import matplotlib.pyplot as plt

    %matplotlib inline
    np.random.seed(42)

    pop_data = np.random.gamma(1,100,3000)
    plt.hist(pop_data);

    means_size_100 = []
    for _ in range(10000):
        sample = np.random.choice(pop_data, 100)
        means_size_100.append(sample.mean())
        
    plt.hist(means_size_100);
    ```

# [Bootstrapping](https://towardsdatascience.com/bootstrapping-statistics-what-it-is-and-why-its-used-e2fa29577307)
- Bootstrapping is random sampling with replacement. 
- Bootstrapping is a statistical procedure that resamples a single dataset to create many simulated samples. 
- This process allows for the calculation of standard errors, confidence intervals, and hypothesis testing

How does it work?
- A sample of size n is drawn from the population
- Let us call this sample ***S***. 
- S should be representative of the population. 
- The sampling distribution is created by resampling observations with replacement from S, ***m times***, with each resampled set having ***n observations***. 
- Therefore, by resampling S m times with replacement, it would be as if m samples were drawn from the original population, and the estimates derived would be representative of the theoretical distribution under the traditional approach. It must be noted that increasing the number of resamples, m, will not increase the amount of information in the data. 
    
    ![image10]

- Open notebook under ```notebooks/Building Confidence Intervals.ipynb.ipynb```
    ```
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    np.random.seed(42)

    coffee_full = pd.read_csv('../data/coffee_dataset.csv')
    coffee_red = coffee_full.sample(200) #this is the only data you might actually get in the real world.

    # Simulate 200 "new" individuals from your original sample of 200
    bootsamp = coffee_red.sample(200, replace = True)

    # What are the proportion of coffee drinkers in your bootstrap sample? 
    bootsamp['drinks_coffee'].mean() # Drink Coffee and 1 minus gives those who don't
    ```

    ```
    # Simulate a bootstrap sample 10,000 times
    # take the mean height of the non-coffee drinkers in each sample
    boot_means = []
    for _ in range(10000):
        bootsamp = coffee_red.sample(200, replace = True)
        boot_mean = bootsamp[bootsamp['drinks_coffee'] == False]['height'].mean()
        boot_means.append(boot_mean)
        
    plt.hist(boot_means); # Looks pretty normal
    ```

- Using ***random.choice*** in python actually samples in this way. Where the probability of any number in our set stays the same regardless of how many times it has been chosen. Flipping a coin and rolling a die are kind of like bootstrap sampling as well, as rolling a 6 in one scenario doesn't mean that 6 is less likely later. 

# Confidence intervals
- We can use bootstrapping and sampling distributions to build confidence intervals for our parameters of interest.

    ![image11]

    ![image12]

    ![image13]

- Open notebook under ```notebooks/Building Confidence Intervals.ipynb```

    ```
    np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
    ```
- Check the code under Bootsrapping section (the confidence code line will follow on that code)

- The following example considers confidence intervals for the difference in means. This is similar to the Bootstrpping code above.
- Open notebook under ```notebooks/Confidence Intervals - Difference in Means.ipynb```
    ```
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    %matplotlib inline
    np.random.seed(42)

    full_data = pd.read_csv('../data/coffee_dataset.csv')
    sample_data = full_data.sample(200)

    diffs = []
    for _ in range(10000):
        bootsamp = sample_data.sample(200, replace = True)
        coff_mean = bootsamp[bootsamp['drinks_coffee'] == True]['height'].mean()
        nocoff_mean = bootsamp[bootsamp['drinks_coffee'] == False]['height'].mean()
        diffs.append(coff_mean - nocoff_mean)
        
    np.percentile(diffs, 0.5), np.percentile(diffs, 99.5) 
    # statistical evidence coffee drinkers are on average taller

    diffs_coff_under21 = []
    for _ in range(10000):
        bootsamp = sample_data.sample(200, replace = True)
        under21_coff_mean = bootsamp.query("age == '<21' and drinks_coffee == True")['height'].mean()
        under21_nocoff_mean = bootsamp.query("age == '<21' and drinks_coffee == False")['height'].mean()
        diffs_coff_under21.append(under21_nocoff_mean - under21_coff_mean)
        
    np.percentile(diffs_coff_under21, 2.5), np.percentile(diffs_coff_under21, 97.5)
    # For the under21 group, we have evidence that the non-coffee drinkers are on average taller

    diffs_coff_over21 = []
    for _ in range(10000):
        bootsamp = sample_data.sample(200, replace = True)
        over21_coff_mean = bootsamp.query("age != '<21' and drinks_coffee == True")['height'].mean()
        over21_nocoff_mean = bootsamp.query("age != '<21' and drinks_coffee == False")['height'].mean()
        diffs_coff_over21.append(over21_nocoff_mean - over21_coff_mean)
        
    np.percentile(diffs_coff_over21, 2.5), np.percentile(diffs_coff_over21, 97.5)
    # For the over21 group, we have evidence that on average the non-coffee drinkers are taller
    ```
- ***Confidence level***: e.g. 95% or, 99%
- Assuming you control all other items of your analysis:
    - Increasing your sample size will decrease the width of your confidence interval.
    - Increasing your confidence level (say 95% to 99%) will increase the width of your confidence interval.

- ***Confidence interval width*** is the difference between your upper and lower bounds of your confidence interval
- ***Margin of error***: is half the confidence interval width, and the value that you add and subtract from your sample estimate to achieve your confidence interval final results.

# Confidence Intervals (& Hypothesis Testing) vs. Machine Learning
- Confidence intervals take an aggregate approach towards the conclusions made based on data, as these tests are aimed at understanding population parameters (which are aggregate population values).

- Alternatively, machine learning techniques take an individual approach towards making conclusions, as they attempt to predict an outcome for each specific data point. 

# Practical and Statistical Significance
- Using confidence intervals and hypothesis testing, you are able to provide statistical significance in making decisions.
- However, it is also important to take into consideration practical significance in making decisions. Practical significance takes into consideration other factors of your situation that might not be considered directly in the results of your hypothesis test or confidence interval. Constraints like space, time, or money are important in business decisions. However, they might not be accounted for directly in a statistical test.

# Bootstrapping + Sampling Distributions or Hypothesis Testing?
- Bootstrpping + Sampling Distributions is a powerful method for builing confidence intervals for essentially any parameters we might be interested in.
- Bootstrapping can replace t-test, two-sample t-test, paired t-test, z-test, chi-squared test, f-test
- If you truly believe that your data is representative of the population dataset, the bootstrapping method should provide better confidence intervall results
- However: With large enough sample sizes Bootstrapping and Traditional Methods will provide essentially the same result.

# [Hypothesis testing](https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce)
## What is hypothesis testing?

- A hypothesis is “an idea that can be tested”. Hypothesis testing is a statistical method used for making decisions based on experimental data. It's basically an assumption that we make about the population parameter.
- Hypothesis Testing and Confidence Intervals allow us to use only sample data to draw conclusions about an entire population



## What are the basics of hypothesis testing?

- The basic of hypothesis is [normalisation](https://en.wikipedia.org/wiki/Normalization_(statistics)) and [standard normalisation](https://stats.stackexchange.com/questions/10289/whats-the-difference-between-normalization-and-standardization). All hypothesis tests are based on these 2 terms.

    ![image16]


- ***Normal Distribution***
    1. mean = median = mode
    2. Transformation:

        <img src="https://render.githubusercontent.com/render/math?math=X_{new}=\frac{x - x_{min}}{x_{max} - x_{min}}" width="280px">

- ***Standardised Normal Distribution***

    1. mean = 0 and standard deviation  = 1
    2. Transformation:

        <img src="https://render.githubusercontent.com/render/math?math=X_{new}=\frac{x - \mu}{\sigma}" width="200px">

## What are important parameters of hypothesis testing?

- ***Null hypothesis vs. Alternate hypotheis***: 

    ![image14]

- Innocent until proven guilty:
    - H0: Innocent, i.e. is true before we collect any data
    - H1: Guilty, i.e. an individual is guilty

- Example

    - <img src="https://render.githubusercontent.com/render/math?math=H_{0}: \mu_{1} = \mu_{2}" width="200px">


    - <img src="https://render.githubusercontent.com/render/math?math=H_{1}: \mu_{1} \neq \mu_{2}" width="200px">

- ***Level of significance***: The probability of rejecting a null hypothesis that is true; the probability of making this error.

    Common significance levels: 0.10,  0.05, 0.01

- ***Statistical errors***
    In general, there are two types of errors we can make while testing: Type I error (False positive) and Type II Error (False negative).

- ***Type I error***: When we reject the null hypothesis, although that hypothesis was true. The probability of committing Type I error (False positive) is equal to the significance level (α).

- ***Type II errors***: When we accept the null hypothesis but it is false. The probability of committing Type II error (False negative) is equal to the beta (β).


    ![image32]

- ***One-sided test***: Used when the null doesn’t contain equality or inequality sign. It contains:
    - ```<``` 
    - ```>```
    - ```≤```
    - ```≥```

    

- ***Two-sided test***: Used when the null contains an 
    - equality ```=```
    - or an inequality sign ```≠```
    

- ***Degree of freedom***: Degrees of Freedom refers to the maximum number of logically independent values, which are values that have the freedom to vary, in the data sample. 

    Example: dataset with 10 values

    - no calculation: 10 degrees of freedom (each datapoint is free to choose)
    - with an estimation (e.g. mean) - one constraint -> sum_total = 10 x mean 
    
## Simulating a sampling distribution from the Null Hypothesis
- In the sectionon confidence intervals, we saw how we could simulate a sampling distribution for a statistic by bootstrapping the sample data. Alternatively, in hypothesis testing, we could simulate a sampling distribution from the null hypothesis using characteristics that would be true if our data were to have come from the null.

- Open notebook under ```notebooks/Simulating From the Null.ipynb```
    ```
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    %matplotlib inline
    np.random.seed(42)

    full_data = pd.read_csv('coffee_dataset.csv')
    sample_data = full_data.sample(200)

    nocoff_means, coff_means, diffs = [], [], []

    for _ in range(10000):
        bootsamp = sample_data.sample(200, replace = True)
        coff_mean = bootsamp[bootsamp['drinks_coffee'] == True]['height'].mean()
        nocoff_mean = bootsamp[bootsamp['drinks_coffee'] == False]['height'].mean()
        # append the info 
        coff_means.append(coff_mean)
        nocoff_means.append(nocoff_mean)
        diffs.append(coff_mean - nocoff_mean)   

    np.std(nocoff_means) # the standard deviation of the sampling distribution for nocoff

    np.std(coff_means) # the standard deviation of the sampling distribution for coff

    np.std(diffs) # the standard deviation for the sampling distribution for difference in means

    plt.hist(nocoff_means, alpha = 0.5);
    plt.hist(coff_means, alpha = 0.5); # They look pretty normal to me!
    plt.hist(diffs, alpha = 0.5); # again normal - this is by the central limit theorem

    # Simulate a sampling distribution from the null hypothesis
    null_vals = np.random.normal(0, np.std(diffs), 10000) # Here are 10000 draws from the sampling distribution under the null
    ```
## P-value 
- p-value is the conditional probability of observing your statistic if the null hypothesis is true. P(statistic | H0 = True)
- The p-value is the smallest level of significance at which we can still reject the null hypothesis
- If p-value is less than the chosen significance level then you reject the null hypothesis i.e. you accept  alternative hypothesis.
- Here is a link to a [p_value claculator](https://www.socscistatistics.com/pvalues/)

    ![image27]

- Open notebook under ```notebooks/Simulating From the Null.ipynb``` (see the code above and add the following lines)

    ```
    null_vals = np.random.normal(70, np.std(coff_means), 10000)
    plt.hist(null_vals); #Here is the sampling distribution of coff_means
    sample_mean = sample_data.height.mean()

    # p-value calculation
    (null_vals > sample_mean).mean()
    # Result: accept the Null

    # p-value calculation
    (null_vals < sample_mean).mean()
    # Result: reject the Null

    null_mean = 70
    (null_vals < sample_mean).mean() + (null_vals > null_mean + (null_mean - sample_mean)).mean()
    # Result: reject the Null
    ```

## Conclusions in Hypothesis Testing - Calculating errors
- The word ***accept*** is one that is avoided when making statements regarding the null and alternative. You are not stating that one of the hypotheses is true. Rather, you are making a decision based on the likelihood of your data coming from the null hypothesis with regard to your type I error threshold.

- Therefore, the wording used in conclusions of hypothesis testing includes: We reject the null hypothesis or We fail to reject the null hypothesis. This lends itself to the idea that you start with the null hypothesis true by default, and "choosing" the null at the end of the test would have been the choice even if no data were collected.


    ![image35]

     ![image29]

- Open notebook under ```notebooks/Drawing Conclusions.ipynb```

    ```
    import numpy as np
    import pandas as pd

    jud_data = pd.read_csv('judicial_dataset_predictions.csv')
    par_data = pd.read_csv('parachute_dataset.csv')

    jud_data[jud_data['actual'] != jud_data['predicted']].shape[0]/jud_data.shape[0] # Number of errors
    jud_data.query("actual == 'innocent' and predicted == 'guilty'").count()[0]/jud_data.shape[0] # Type 1 errors
    jud_data.query("actual == 'guilty' and predicted == 'innocent'").count()[0]/jud_data.shape[0] # Type 2 errors

    # If everyone was predicted to be guilty, then every actual innocent 
    # person would be a type I error.
    # Type I = pred guilty, but actual = innocent
    jud_data[jud_data['actual'] == 'innocent'].shape[0]/jud_data.shape[0]

    #If everyone has prediction of guilty, then no one is predicted inncoent
    #Therefore, there would be no type 2 errors in this case
    # Type II errs = pred innocent, but actual = guilty
    0
    ```



## Impact of Large Sample Size
- With large sample sizes, hypothesis testing leads to even the smallest of findings as statistically significant. However, these findings might not be practically significant at all. 
- Alternatively, machine learning techniques take an individual approach towards making conclusions, as they attempt to predict an outcome for each specific data point. 

    ![image36]

- Open notebook under ```notebooks/What is the impact of sample size.ipynb```

## How do Confidence Intervals and Hypothesis Testing compare?
- A two-sided hypothesis test (that is a test involving a ≠ in the alternative) is the same in terms of the conclusions made as a confidence interval as long as:

    1 − CI = α 

- For example, a 95% confidence interval will draw the same conclusions as a hypothesis test with a type I error rate of 0.05 in terms of which hypothesis to choose, because:

    1 − 0.95 = 0.05

    assuming that the alternative hypothesis is a two sided test.

    ![image37]

## Multiple Tests
- When performing multiple hypothesis tests, your errors will compound. Therefore, using some sort of correction to maintain your true Type I error rate is important. A simple, but very conservative approach is to use what is known as a ***Bonferroni correction***, which says you should just divide your α level (or Type I error threshold) by the number of tests performed.

- Open notebook under ```notebooks/Multiple Testing.ipynb```

## Summary
- Short overview of important (practical) terms in ference statistics

    ![image38]


## Hypothesis testing types
- T Test ( Student T test)
- Z Test
- ANOVA Test
- Chi-Square Test

    ![image28]

- ***Python SciPy to test for Hypothesis Testing***
    - SciPy [ttest_1samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)
        ```
        scipy.stats.ttest_1samp(
            a, # array like input
            popmean, # Expected value in null hypothesis (float or array_like)
            axis=0, # Axis along which to compute test; default is 0. If None, compute over the whole array a.
            nan_policy='propagate', # Defines how to handle when input contains nan
            alternative='two-sided') # 'less' -> one-sided, 'greater'  -> one-sided
        ```

- ***T- Test***:
    - ***One sample - two sided - Student's t-test***: determines whether the sample mean is statistically different from a known or hypothesised population mean.

        Example : You have 10 ages and you are checking whether avg age is 30 or not. 

        - <img src="https://render.githubusercontent.com/render/math?math=H_{0}: \mu = 30" width="100px">
        
        - <img src="https://render.githubusercontent.com/render/math?math=H_{1}: \mu \neq 30" width="100px">

        ```
        # One sample - two sided - Student's t-test
        from scipy.stats import ttest_1samp
        import numpy as np

        ages = np.genfromtxt(“ages.csv”)

        print(ages)
        ages_mean = np.mean(ages)
        print(ages_mean)
        tset, pval = ttest_1samp(ages, 30)

        print(“p-values”, pval)

        if pval < 0.05:    # alpha value is 0.05 or 5%
            print("reject null hypothesis")
        else:
            print("accept null hypothesis")
        ```

    - ***Two (independent) samples - two sided - Student's t-test***: compares the means of two independent groups in order to determine whether there is statistical evidence that the associated population means are significantly different.

        Example : is there any association between data1 and data2 

        - <img src="https://render.githubusercontent.com/render/math?math=H_{0}: \mu_{1} = \mu_{2}" width="100px">
        - <img src="https://render.githubusercontent.com/render/math?math=H_{1}: \mu_{1} \neq \mu_{2}" width="100px">

        ```
        # Two (independent) samples - two sided - Student's t-test
        from scipy.stats import ttest_ind

        data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
        data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
        stat, p = ttest_ind(data1, data2)
        print('stat=%.3f, p=%.3f' % (stat, p))

        if pval<0.05:
            print("reject null hypothesis")
        else:
            print("accept null hypothesis")
        ```
    - ***Two (dependent/paired) samples - two sided - Student's t-test***: Test for a significant difference between 2 related variables. An example of this is if you where to collect the blood pressure for an individual before and after some treatment, condition, or time point.

        - <img src="https://render.githubusercontent.com/render/math?math=H_{0}: \mu_{1} - \mu_{2} = 0" width="150px">
        - <img src="https://render.githubusercontent.com/render/math?math=H_{1}: \mu_{1} - \mu_{2} \neq 0" width="150px">

        ```
        # Two (dependent/paired) samples - two sided - Student's t-test
        from scipy.stats import ttest_rel
        data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
        data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
        stat, p = ttest_rel(data1, data2)
        print('stat=%.3f, p=%.3f' % (stat, p))

        if pval<0.05:
            print("reject null hypothesis")
        else:
            print("accept null hypothesis")
        ```

- ***When should you run a Z Test?***

    You would use a Z test if:

    - Your sample size is greater than 30. Otherwise, use a t test.
    - Data points should be independent from each other. In other words, one data point isn’t related or doesn’t affect another data point.
    - Your data should be normally distributed. However, for large sample sizes (over 30) this doesn’t always matter.
    - Your data should be randomly selected from a population, where each item has an equal chance of being selected.
    - Sample sizes should be equal if at all possible.

    - Use [statsmodels ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html)

    - ***One-sample - two sided - z-test***

        Example: Again we are using blood pressure with some mean like 156 for z-test.

        - <img src="https://render.githubusercontent.com/render/math?math=H_{0}: \mu = 156" width="100px">
        - <img src="https://render.githubusercontent.com/render/math?math=H_{1}: \mu \neq 156" width="100px">

        ```
        # One-sample - two sided - z-test
        import pandas as pd
        from scipy import stats
        from statsmodels.stats import weightstats as stests

        ztest, pval = stests.ztest(df['bp_before'], x2=None, value=156)
        print(float(pval))

        if pval<0.05:
            print("reject null hypothesis")
        else:
            print("accept null hypothesis")
        ```

    - ***Two (independent) samples - two sided - z-test***
        In two sample z-test , similar to t-test here we are checking ***two independent data groups*** and deciding whether sample mean of two groups is equal or not.

        - <img src="https://render.githubusercontent.com/render/math?math=H_{0}: \mu_{1} = \mu_{2}" width="100px">
        - <img src="https://render.githubusercontent.com/render/math?math=H_{1}: \mu_{1} \neq \mu_{2}" width="100px">

        Example : we are checking in blood data after blood and before blood data.(code in python below)

        ```
        # Two (independent) samples - two sided - z-test
        ztest, pval1 = stests.ztest(df['bp_before'], x2=df['bp_after'], value=0, alternative='two-sided')

        print(float(pval1))

        if pval<0.05:
            print("reject null hypothesis")
        else:
            print("accept null hypothesis")
        ```

- ***ANOVA*** (F-Test): The t-test works well when dealing with two groups, but sometimes we want to compare more than two groups at the same time. The analysis of variance or ANOVA is a statistical inference test that lets you compare multiple groups at the same time.

    F = Between group variability / Within group variability

    ![image21]

    - ***One Way F-Test***: It tells whether two or more groups are similar or not based on their mean similarity and f-score.

        Example : there are 3 different categories of plants and their weights and need to check whether all 3 groups are similar or not

        ```
        df_anova = pd.read_csv('PlantGrowth.csv')
        df_anova = df_anova[['weight','group']]

        grps = pd.unique(df_anova.group.values)
        d_data = {grp:df_anova['weight'][df_anova.group == grp] for grp in grps}

        F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])

        print("p-value for significance is: ", p)

        if p<0.05:
            print("reject null hypothesis")
        else:
            print("accept null hypothesis")
        ```

    - ***Two Way F-test*** : Two way F-test is extension of 1-way f-test, it is used when we have 2 independent variable and 2+ groups. 2-way F-test does not tell which variable is dominant. If we need to check individual significance then Post-hoc testing need to be performed.


        Now let’s take a look at the Grand mean crop yield (the mean crop yield not by any sub-group), as well the mean crop yield by each factor, as well as by the factors grouped

        ```
        import statsmodels.api as sm
        from statsmodels.formula.api import olsdf_anova2 = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/crop_yield.csv")

        model = ols('Yield ~ C(Fert)*C(Water)', df_anova2).fit()
        print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

        res = sm.stats.anova_lm(model, typ= 2)
        res
        ```
- ***Chi-Square Test*** is applied when you have two categorical variables from a single population. It is used to determine whether there is a significant association between the two variables.

    For example, in an election survey, voters might be classified by gender (male or female) and voting preference (Democrat, Republican, or Independent). We could use a chi-square test for independence to determine whether gender is related to voting preference

    ```
    df_chi = pd.read_csv('chi-test.csv')
    contingency_table=pd.crosstab(df_chi["Gender"],df_chi["Shopping?"])
    print('contingency_table :-\n',contingency_table)

    #Observed Values
    Observed_Values = contingency_table.values
    print("Observed Values :-\n",Observed_Values)

    b=stats.chi2_contingency(contingency_table)
    Expected_Values = b[3]
    print("Expected Values :-\n",Expected_Values)

    no_of_rows=len(contingency_table.iloc[0:2,0])
    no_of_columns=len(contingency_table.iloc[0,0:2])
    ddof=(no_of_rows-1)*(no_of_columns-1)
    print("Degree of Freedom:-",ddof)
    alpha = 0.05

    from scipy.stats import chi2
    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
    chi_square_statistic=chi_square[0]+chi_square[1]
    print("chi-square statistic:-",chi_square_statistic)

    critical_value=chi2.ppf(q=1-alpha,df=ddof)
    print('critical_value:',critical_value)

    #p-value
    p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
    print('p-value:',p_value)


    print('Significance level: ',alpha)
    print('Degree of Freedom: ',ddof)
    print('chi-square statistic:',chi_square_statistic)
    print('critical_value:',critical_value)
    print('p-value:',p_value)

    if chi_square_statistic>=critical_value:
        print("Reject H0,There is a relationship between 2 categorical variables")
    else:
        print("Retain H0,There is no relationship between 2 categorical variables")

    if p_value<=alpha:
        print("Reject H0,There is a relationship between 2 categorical variables")
    else:
        print("Retain H0,There is no relationship between 2 categorical variables")
    ```
















## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- If you need a Command Line Interface (CLI) under Windows you could use [git](https://git-scm.com/). Under Mac OS use the pre-installed Terminal.

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Disaster-Response-Pipeline-Project.git
```

- Change Directory
```
$ cd Disaster-Response-Pipeline-Project
```

- Create a new Python environment, e.g. ds_ndp. Inside Git Bash (Terminal) write:
```
$ conda create --name ds_ndp
```

- Activate the installed environment via
```
$ conda activate ds_ndp
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
scikit-learn = 0.20
pipelinehelper = 0.7.8
```
Example via pip:
```
pip install numpy
pip install pandas
pip install scikit-learn==0.20
pip install pipelinehelper
```
scikit-learn==0.20 is needed for sklearns dictionary output (output_dict=True) for the classification_report. Earlier versions do not support this.


Link1 to [pipelinehelper](https://github.com/bmurauer/pipelinehelper)

Link2 to [pipelinehelper](https://stackoverflow.com/questions/23045318/scikit-grid-search-over-multiple-classifiers)

- Check the environment installation via
```
$ conda env list
```

### Switch the pipelines
- Active pipeline at the moment: pipeline_1 (Fast training/testing) pipeline
- More sophisticated pipelines start with pipeline_2.
- Model training has been done with ```pipeline_2```.
- In order to switch between pipelines or to add more pipelines open train.classifier.py. Adjust the pipelines in `def main()` which should be tested (only one or more are possible) via the list ```pipeline_names```.

```
def main():
   ...

    if len(sys.argv) == 3:
        ...

        # start pipelining, build the model
        pipeline_names = ['pipeline_1', 'pipeline_2']
```


### Run the web App

1. Run the following commands in the project's root directory to set up your database and model.

    To run ETL pipeline that cleans data and stores in database


        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    To run ML pipeline that trains classifier and saves

        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


2. Run the following command in the app's directory to run your web app

        python run.py




3. Go to http://0.0.0.0:3001/

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
## Links
* [Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)
* [17 Statistical Hypothesis Tests in Python ](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)
* [What is Sampling Distribution](https://www.slideshare.net/DonnaWiles1/sampling-distribution-84492010)
* [Bootstrapping Statistics. What it is and why it’s used.](https://towardsdatascience.com/bootstrapping-statistics-what-it-is-and-why-its-used-e2fa29577307)

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Data Visualization
* [10 Python Data Visualization Libraries for Any Field | Mode](https://mode.com/blog/python-data-visualization-libraries/)
* [5 Quick and Easy Data Visualizations in Python with Code](https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f)
* [The Best Python Data Visualization Libraries](https://www.fusioncharts.com/blog/best-python-data-visualization-libraries/)