
# Predicting sentiment from product reviews


The goal of this first notebook is to explore logistic regression and feature engineering with existing GraphLab functions.

In this notebook you will use product review data from Amazon.com to predict whether the sentiments about a product (from its reviews) are positive or negative.

* Use SFrames to do some feature engineering
* Train a logistic regression model to predict the sentiment of product reviews.
* Inspect the weights (coefficients) of a trained logistic regression model.
* Make a prediction (both class and probability) of sentiment for a new product review.
* Given the logistic regression weights, predictors and ground truth labels, write a function to compute the **accuracy** of the model.
* Inspect the coefficients of the logistic regression model and interpret their meanings.
* Compare multiple logistic regression models.

Let's get started!
    
## Fire up GraphLab Create

Make sure you have the latest version of GraphLab Create.


```python
from __future__ import division
import graphlab
import math
import string
```

# Data preperation

We will use a dataset consisting of baby product reviews on Amazon.com.


```python
products = graphlab.SFrame('amazon_baby.gl/')
```

    [INFO] GraphLab Create v1.8.3 started. Logging: C:\Users\MANOJK~1\AppData\Local\Temp\graphlab_server_1457214773.log.0
    

Now, let us see a preview of what the dataset looks like.


```python
products
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Flannel Wipes</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">These flannel wipes are<br>OK, but in my opinion ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Wipe Pouch</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">it came early and was not<br>disappointed. i love ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Annas Dream Full Quilt<br>with 2 Shams ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Very soft and comfortable<br>and warmer than it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This is a product well<br>worth the purchase.  I ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">All of my kids have cried<br>non-stop when I tried to ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">When the Binky Fairy came<br>to our house, we didn't ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A Tale of Baby's Days<br>with Peter Rabbit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lovely book, it's bound<br>tightly so you may no ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Perfect for new parents.<br>We were able to keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A friend of mine pinned<br>this product on Pinte ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
    </tr>
</table>
[183531 rows x 3 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Build the word count vector for each review

Let us explore a specific example of a baby product.



```python
products[269]
```




    {'name': 'The First Years Massaging Action Teether',
     'rating': 5.0,
     'review': 'A favorite in our house!'}



Now, we will perform 2 simple data transformations:

1. Remove punctuation using [Python's built-in](https://docs.python.org/2/library/string.html) string functionality.
2. Transform the reviews into word-counts.

**Aside**. In this notebook, we remove all punctuations for the sake of simplicity. A smarter approach to punctuations would preserve phrases such as "I'd", "would've", "hadn't" and so forth. See [this page](https://www.cis.upenn.edu/~treebank/tokenization.html) for an example of smart handling of punctuations.


```python
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

review_without_puctuation = products['review'].apply(remove_punctuation)
products['word_count'] = graphlab.text_analytics.count_words(review_without_puctuation)
```

Now, let us explore what the sample example above looks like after these 2 transformations. Here, each entry in the **word_count** column is a dictionary where the key is the word and the value is a count of the number of times the word occurs.


```python
products[269]['word_count']
```




    {'a': 1L, 'favorite': 1L, 'house': 1L, 'in': 1L, 'our': 1L}



## Extract sentiments

We will **ignore** all reviews with *rating = 3*, since they tend to have a neutral sentiment.


```python
products = products[products['rating'] != 3]
len(products)
```




    166752



Now, we will assign reviews with a rating of 4 or higher to be *positive* reviews, while the ones with rating of 2 or lower are *negative*. For the sentiment column, we use +1 for the positive class label and -1 for the negative class label.


```python
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Wipe Pouch</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">it came early and was not<br>disappointed. i love ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'love': 1L,<br>'it': 3L, 'highly': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Annas Dream Full Quilt<br>with 2 Shams ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Very soft and comfortable<br>and warmer than it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'quilt': 1L,<br>'it': 1L, 'comfortable': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This is a product well<br>worth the purchase.  I ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'ingenious':<br>1L, 'love': 2L, 'is': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">All of my kids have cried<br>non-stop when I tried to ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'all': 2L,<br>'help': 1L, 'cried': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">When the Binky Fairy came<br>to our house, we didn't ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'cute': 1L,<br>'help': 2L, 'habit': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A Tale of Baby's Days<br>with Peter Rabbit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lovely book, it's bound<br>tightly so you may no ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'shop': 1L, 'be': 1L,<br>'is': 1L, 'bound': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Perfect for new parents.<br>We were able to keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'all': 1L,<br>'right': 1L, 'able': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A friend of mine pinned<br>this product on Pinte ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'fantastic':<br>1L, 'help': 1L, 'give': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'standarad':<br>1L, 'another': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this journal and<br>our nanny uses it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 2L, 'nannys': 1L,<br>'just': 1L, 'sleep': 2L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[166752 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Now, we can see that the dataset contains an extra column called **sentiment** which is either positive (+1) or negative (-1).

## Split data into training and test sets

Let's perform a train/test split with 80% of the data in the training set and 20% of the data in the test set. We use `seed=1` so that everyone gets the same result.


```python
train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)
```

    133416
    33336
    

# Train a sentiment classifier with logistic regression

We will now use logistic regression to create a sentiment classifier on the training data. This model will use the column **word_count** as a feature and the column **sentiment** as the target. We will use `validation_set=None` to obtain same results as everyone else.

**Note:** This line may take 1-2 minutes.


```python
sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target = 'sentiment',
                                                      features=['word_count'],
                                                      validation_set=None)
```


```python
sentiment_model
```




    Class                         : LogisticClassifier
    
    Schema
    ------
    Number of coefficients        : 121713
    Number of examples            : 133416
    Number of classes             : 2
    Number of feature columns     : 1
    Number of unpacked features   : 121712
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.01
    
    Training Summary
    ----------------
    Solver                        : auto
    Solver iterations             : 6
    Solver status                 : TERMINATED: Terminated due to numerical difficulties.
    Training time (sec)           : 8.5671
    
    Settings
    --------
    Log-likelihood                : inf
    
    Highest Positive Coefficients
    -----------------------------
    word_count[mobileupdate]      : 41.9847
    word_count[placeid]           : 41.7354
    word_count[labelbox]          : 41.151
    word_count[httpwwwamazoncomreviewrhgg6qp7tdnhbrefcmcrprcmtieutf8asinb00318cla0nodeid]: 40.0454
    word_count[knobskeeping]      : 36.2091
    
    Lowest Negative Coefficients
    ----------------------------
    word_count[probelm]           : -44.9283
    word_count[impulsejeep]       : -43.081
    word_count[infantsyoung]      : -39.5945
    word_count[cutereditafter]    : -35.6875
    word_count[avacado]           : -35.0542



**Aside**. You may get an warning to the effect of "Terminated due to numerical difficulties --- this model may not be ideal". It means that the quality metric (to be covered in Module 3) failed to improve in the last iteration of the run. The difficulty arises as the sentiment model puts too much weight on extremely rare words. A way to rectify this is to apply regularization, to be covered in Module 4. Regularization lessens the effect of extremely rare words. For the purpose of this assignment, however, please proceed with the model above.

Now that we have fitted the model, we can extract the weights (coefficients) as an SFrame as follows:


```python
weights = sentiment_model.coefficients
weights.column_names()
```




    ['name', 'index', 'class', 'value', 'stderr']



There are a total of `121713` coefficients in the model. Recall from the lecture that positive weights $w_j$ correspond to weights that cause positive sentiment, while negative weights correspond to negative sentiment. 

Fill in the following block of code to calculate how many *weights* are positive ( >= 0). (**Hint**: The `'value'` column in SFrame *weights* must be positive ( >= 0)).


```python
weights['value'].head()
```




    dtype: float
    Rows: 10
    [1.3033708054359998, 0.303815600015292, 1.4918301527570792, -3.9574861839267452, 1.4330168543924398, 0.009866464903066302, -0.79776455392579, 0.048449573171956806, 0.16554143661471077, -0.09499379472684794]




```python
num_positive_weights = weights[weights['value'] >= 0]
print len(num_positive_weights)
```

    68419
    


```python
num_negative_weights = weights[weights['value'] < 0]
print len(num_negative_weights)
```

    53294
    


```python
num_positive_weights = weights[weights['value'] >= 0]
num_negative_weights = weights[weights['value'] < 0]

print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights
```

    Number of positive weights: +-------------+-----------+-------+------------------+--------+
    |     name    |   index   | class |      value       | stderr |
    +-------------+-----------+-------+------------------+--------+
    | (intercept) |    None   |   1   |  1.30337080544   |  None  |
    |  word_count | recommend |   1   |  0.303815600015  |  None  |
    |  word_count |   highly  |   1   |  1.49183015276   |  None  |
    |  word_count |    love   |   1   |  1.43301685439   |  None  |
    |  word_count |     it    |   1   | 0.00986646490307 |  None  |
    |  word_count |    and    |   1   |  0.048449573172  |  None  |
    |  word_count |    bags   |   1   |  0.165541436615  |  None  |
    |  word_count |   early   |   1   |  0.488413478808  |  None  |
    |  word_count |    came   |   1   |  0.131378480765  |  None  |
    |  word_count |     i     |   1   | 0.0182528116279  |  None  |
    +-------------+-----------+-------+------------------+--------+
    [? rows x 5 columns]
    Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
    You can use len(sf) to force materialization. 
    Number of negative weights: +------------+--------------+-------+------------------+--------+
    |    name    |    index     | class |      value       | stderr |
    +------------+--------------+-------+------------------+--------+
    | word_count | disappointed |   1   |  -3.95748618393  |  None  |
    | word_count |    planet    |   1   | -0.797764553926  |  None  |
    | word_count |    wipes     |   1   | -0.0949937947268 |  None  |
    | word_count |     not      |   1   | -0.817151716331  |  None  |
    | word_count |     does     |   1   | -0.0769014155235 |  None  |
    | word_count |     wise     |   1   | -0.302595277297  |  None  |
    | word_count |     was      |   1   | -0.0530004786379 |  None  |
    | word_count |    holder    |   1   | -0.0300523581012 |  None  |
    | word_count |     leak     |   1   |  -0.24658014554  |  None  |
    | word_count |      of      |   1   | -0.0227274977677 |  None  |
    +------------+--------------+-------+------------------+--------+
    [? rows x 5 columns]
    Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
    You can use len(sf) to force materialization. 
    


```python
print len(num_positive_weights)
```

    68419
    

**Quiz question:** How many weights are >= 0?

## Making predictions with logistic regression

Now that a model is trained, we can make predictions on the **test data**. In this section, we will explore this in the context of 3 examples in the test dataset.  We refer to this set of 3 examples as the **sample_test_data**.


```python
sample_test_data = test_data[10:13]
print sample_test_data['rating']
sample_test_data
```

    [5.0, 2.0, 1.0]
    




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Our Baby Girl Memory Book</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Absolutely love it and<br>all of the Scripture in ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'all': 1L,<br>'love': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Wall Decor Removable<br>Decal Sticker - Colorful ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Would not purchase again<br>or recommend. The decals ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'wall': 1L,<br>'them': 1L, 'decals': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">New Style Trailing Cherry<br>Blossom Tree Decal ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Was so excited to get<br>this product for my baby ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'money': 1L,<br>'into': 1L, 'it': 3L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
</table>
[3 rows x 5 columns]<br/>
</div>



Let's dig deeper into the first row of the **sample_test_data**. Here's the full review:


```python
sample_test_data[0]['review']
```




    'Absolutely love it and all of the Scripture in it.  I purchased the Baby Boy version for my grandson when he was born and my daughter-in-law was thrilled to receive the same book again.'



That review seems pretty positive.

Now, let's see what the next row of the **sample_test_data** looks like. As we could guess from the sentiment (-1), the review is quite negative.


```python
sample_test_data[1]['review']
```




    'Would not purchase again or recommend. The decals were thick almost plastic like and were coming off the wall as I was applying them! The would NOT stick! Literally stayed stuck for about 5 minutes then started peeling off.'



We will now make a **class** prediction for the **sample_test_data**. The `sentiment_model` should predict **+1** if the sentiment is positive and **-1** if the sentiment is negative. Recall from the lecture that the **score** (sometimes called **margin**) for the logistic regression model  is defined as:

$$
\mbox{score}_i = \mathbf{w}^T h(\mathbf{x}_i)
$$ 

where $h(\mathbf{x}_i)$ represents the features for example $i$.  We will write some code to obtain the **scores** using GraphLab Create. For each row, the **score** (or margin) is a number in the range **[-inf, inf]**.


```python
scores = sentiment_model.predict(sample_test_data, output_type='margin')
print scores
```

    [6.734619727058982, -5.734130996760175, -14.668460404468366]
    

### Predicting sentiment

These scores can be used to make class predictions as follows:

$$
\hat{y} = 
\left\{
\begin{array}{ll}
      +1 & \mathbf{w}^T h(\mathbf{x}_i) > 0 \\
      -1 & \mathbf{w}^T h(\mathbf{x}_i) \leq 0 \\
\end{array} 
\right.
$$

Using scores, write code to calculate $\hat{y}$, the class predictions:


```python
def pred(score):
    if score > 0:
        return (1)
    else:
        return (-1)

```


```python
pred(scores[2])
```




    -1



Run the following code to verify that the class predictions obtained by your calculations are the same as that obtained from GraphLab Create.


```python
print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data)
```

    Class predictions according to GraphLab Create:
    [1L, -1L, -1L]
    

**Checkpoint**: Make sure your class predictions match with the one obtained from GraphLab Create.

### Probability predictions

Recall from the lectures that we can also calculate the probability predictions from the scores using:
$$
P(y_i = +1 | \mathbf{x}_i,\mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))}.
$$

Using the variable **scores** calculated previously, write code to calculate the probability that a sentiment is positive using the above formula. For each row, the probabilities should be a number in the range **[0, 1]**.


```python

```

**Checkpoint**: Make sure your probability predictions match the ones obtained from GraphLab Create.


```python
print "Class predictions according to GraphLab Create:" 
print sentiment_model.predict(sample_test_data, output_type='probability')
```

    Class predictions according to GraphLab Create:
    [0.9988123848377194, 0.003223268181800946, 4.261557996656069e-07]
    

** Quiz Question:** Of the three data points in **sample_test_data**, which one (first, second, or third) has the **lowest probability** of being classified as a positive review?

# Find the most positive (and negative) review

We now turn to examining the full test dataset, **test_data**, and use GraphLab Create to form predictions on all of the test data points for faster performance.

Using the `sentiment_model`, find the 20 reviews in the entire **test_data** with the **highest probability** of being classified as a **positive review**. We refer to these as the "most positive reviews."

To calculate these top-20 reviews, use the following steps:
1.  Make probability predictions on **test_data** using the `sentiment_model`. (**Hint:** When you call `.predict` to make predictions on the test data, use option `output_type='probability'` to output the probability rather than just the most likely class.)
2.  Sort the data according to those predictions and pick the top 20. (**Hint:** You can use the `.topk` method on an SFrame to find the top k rows sorted according to the value of a specified column.)


```python
# Scores
scores = sentiment_model.predict(test_data, output_type='margin')
```


```python
scr = graphlab.SFrame(scores)
scr.head()
```


```python
scr = scr.sort('X1', ascending=False)
scr.print_rows(num_rows=20) 
```

    +---------------+
    |       X1      |
    +---------------+
    | 128.601262666 |
    | 93.3205461385 |
    |  80.357507774 |
    | 76.9287951949 |
    | 75.0432136183 |
    |  68.811904249 |
    | 68.3097080032 |
    | 65.4009056213 |
    | 64.5982715178 |
    | 63.6371983848 |
    | 62.1066696134 |
    | 61.9253719891 |
    | 60.9084177668 |
    | 60.5776248308 |
    | 60.0532506467 |
    | 59.9628927332 |
    | 59.2006916976 |
    | 58.9228374559 |
    | 58.8978701016 |
    | 56.7653294597 |
    +---------------+
    [33336 rows x 1 columns]
    
    
# Prediction
ModPred = sentiment_model.predict(test_data)

```python
# Class predictions
CPred = sentiment_model.predict(test_data, output_type='probability')
CPred.head()
```




    dtype: float
    Rows: 10
    [0.7583998877520624, 0.9999999999664292, 0.228950978080386, 0.9999995580631229, 0.9905421692480386, 0.9999992959681095, 0.999764476280349, 0.7228344662831502, 0.999266840896092, 0.9997868300478303]




```python
CPredscr = graphlab.SFrame(CPred)
CPredscr.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">X1</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.758399887752</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999999999966</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.22895097808</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999999558063</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.990542169248</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999999295968</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.99976447628</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.722834466283</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999266840896</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999786830048</td>
    </tr>
</table>
[10 rows x 1 columns]<br/>
</div>




```python
# top-20 reviews
CPredsrt = CPredscr.sort('X1', ascending=False)
CPredsrt.print_rows(num_rows=20) 
```

    +-----+
    |  X1 |
    +-----+
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    | 1.0 |
    +-----+
    [33336 rows x 1 columns]
    
    


```python
test_data.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'standarad':<br>1L, 'another': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this journal and<br>our nanny uses it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 2L, 'nannys': 1L,<br>'just': 1L, 'sleep': 2L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Nature's Lullabies First<br>Year Sticker Calendar ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this little<br>calender, you can keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'babys': 1L,<br>'love': 1L, 'like': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Nature's Lullabies Second<br>Year Sticker Calendar ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I had a hard time finding<br>a second year calendar, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'all': 1L,<br>'months': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lamaze Peekaboo, I Love<br>You ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">One of baby's first and<br>favorite books, and i ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'because':<br>1L, 'family': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lamaze Peekaboo, I Love<br>You ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">My son loved this book as<br>an infant.  It was ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'being': 1L,<br>'infant': 1L, 'course': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lamaze Peekaboo, I Love<br>You ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Our baby loves this book<br>&amp; has loved it for a ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'own': 1L,<br>'it': 3L, 'our': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">SoftPlay Giggle Jiggle<br>Funbook, Happy Bear ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This bear is absolutely<br>adorable and I would  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'cute': 1L,<br>'rating': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">SoftPlay Peek-A-Boo<br>Where's Elmo A Childr ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I bought two for recent<br>baby showers!  The book ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'beautiful': 1L, 'and':<br>2L, 'love': 1L, 'elmo': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby's First Year Undated<br>Wall Calendar with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I searched high and low<br>for a first year cale ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'remembering': 1L,<br>'and': 4L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[10 rows x 5 columns]<br/>
</div>




```python
test_data['scores'] = sentiment_model.predict(test_data, output_type='probability')
```


```python
test_data.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'standarad':<br>1L, 'another': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this journal and<br>our nanny uses it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 2L, 'nannys': 1L,<br>'just': 1L, 'sleep': 2L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Nature's Lullabies First<br>Year Sticker Calendar ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this little<br>calender, you can keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'babys': 1L,<br>'love': 1L, 'like': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Nature's Lullabies Second<br>Year Sticker Calendar ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I had a hard time finding<br>a second year calendar, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'all': 1L,<br>'months': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lamaze Peekaboo, I Love<br>You ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">One of baby's first and<br>favorite books, and i ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2L, 'because':<br>1L, 'family': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lamaze Peekaboo, I Love<br>You ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">My son loved this book as<br>an infant.  It was ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1L, 'being': 1L,<br>'infant': 1L, 'course': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lamaze Peekaboo, I Love<br>You ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Our baby loves this book<br>&amp; has loved it for a ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1L, 'own': 1L,<br>'it': 3L, 'our': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">SoftPlay Giggle Jiggle<br>Funbook, Happy Bear ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This bear is absolutely<br>adorable and I would  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3L, 'cute': 1L,<br>'rating': 1L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">SoftPlay Peek-A-Boo<br>Where's Elmo A Childr ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I bought two for recent<br>baby showers!  The book ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'beautiful': 1L, 'and':<br>2L, 'love': 1L, 'elmo': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby's First Year Undated<br>Wall Calendar with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I searched high and low<br>for a first year cale ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'remembering': 1L,<br>'and': 4L, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">scores</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.758399887752</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999999999966</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.22895097808</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999999558063</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.990542169248</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999999295968</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.99976447628</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.722834466283</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999266840896</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.999786830048</td>
    </tr>
</table>
[10 rows x 6 columns]<br/>
</div>




```python
test1 = test_data.topk('scores', k=20, reverse=False)
```


```python
test1.print_rows(num_rows=20)
```

    +-------------------------------+-------------------------------+--------+
    |              name             |             review            | rating |
    +-------------------------------+-------------------------------+--------+
    | Britax Decathlon Convertib... | I researched a few differe... |  4.0   |
    | Ameda Purely Yours Breast ... | As with many new moms, I c... |  4.0   |
    | Traveling Toddler Car Seat... | I am sure this product wor... |  2.0   |
    | Shermag Glider Rocker Comb... | After searching in stores ... |  4.0   |
    | Cloud b Sound Machine Soot... | First off, I love plush sh... |  5.0   |
    | JP Lizzy Chocolate Ice Cla... | I got this bag as a presen... |  4.0   |
    | Fisher-Price Rainforest Me... | My daughter wasn't able to... |  5.0   |
    | Lilly Gold Sit 'n' Stroll ... | I just completed a two-mon... |  5.0   |
    |  Fisher-Price Deluxe Jumperoo | I had already decided that... |  5.0   |
    | North States Supergate Pre... | I got a couple of these to... |  4.0   |
    |   Munchkin Mozart Magic Cube  | My wife and I have been li... |  4.0   |
    | Britax Marathon Convertibl... | My son began using the Mar... |  5.0   |
    | Wizard Convertible Car Sea... | My son was born big and re... |  5.0   |
    |   Capri Stroller - Red Tech   | First let me say that I wa... |  4.0   |
    | Peg Perego Primo Viaggio C... | We have been using this se... |  5.0   |
    | HALO SleepSack Micro-Fleec... | I love the Sleepsack weara... |  5.0   |
    | Leachco Snoogle Total Body... | I have had my Snoogle for ... |  5.0   |
    | Summer Infant Complete Nur... | This Nursery and Bath Care... |  4.0   |
    | Safety 1st Tot-Lok Four Lo... | I have a wooden desk that ... |  5.0   |
    |  BABYBJORN Potty Chair - Red  | Our family is just startin... |  5.0   |
    +-------------------------------+-------------------------------+--------+
    +-------------------------------+-----------+--------+
    |           word_count          | sentiment | scores |
    +-------------------------------+-----------+--------+
    | {'all': 1L, 'toted': 1L, '... |     1     |  1.0   |
    | {'lansinoh': 1L, 'six': 1L... |     1     |  1.0   |
    | {'all': 1L, 'just': 1L, 'p... |     -1    |  1.0   |
    | {'spotclean': 1L, 'code': ... |     1     |  1.0   |
    | {'cute': 1L, 'all': 2L, 's... |     1     |  1.0   |
    | {'knot': 1L, 'just': 4L, '... |     1     |  1.0   |
    | {'cute': 1L, 'all': 3L, 's... |     1     |  1.0   |
    | {'son': 2L, 'infant': 1L, ... |     1     |  1.0   |
    | {'all': 1L, 'just': 1L, 's... |     1     |  1.0   |
    | {'gates': 5L, 'just': 1L, ... |     1     |  1.0   |
    | {'rating': 1L, 'all': 2L, ... |     1     |  1.0   |
    | {'infant': 1L, 'over': 2L,... |     1     |  1.0   |
    | {'son': 4L, 'all': 1L, 'de... |     1     |  1.0   |
    | {'son': 3L, 'longer': 1L, ... |     1     |  1.0   |
    | {'all': 1L, 'just': 2L, 'm... |     1     |  1.0   |
    | {'all': 1L, 'just': 2L, 'f... |     1     |  1.0   |
    | {'saying': 1L, 'help': 1L,... |     1     |  1.0   |
    | {'son': 1L, 'set': 2L, 'he... |     1     |  1.0   |
    | {'saying': 1L, 'all': 1L, ... |     1     |  1.0   |
    | {'managed': 1L, 'just': 3L... |     1     |  1.0   |
    +-------------------------------+-----------+--------+
    [20 rows x 6 columns]
    
    

**Quiz Question**: Which of the following products are represented in the 20 most positive reviews? [multiple choice]


Now, let us repeat this excercise to find the "most negative reviews." Use the prediction probabilities to find the  20 reviews in the **test_data** with the **lowest probability** of being classified as a **positive review**. Repeat the same steps above but make sure you **sort in the opposite order**.


```python
test2 = test_data.topk('scores', k=20, reverse=True)
test2.print_rows(num_rows=20)
```

    +-------------------------------+-------------------------------+--------+
    |              name             |             review            | rating |
    +-------------------------------+-------------------------------+--------+
    | Jolly Jumper Arctic Sneak ... | I am a "research-aholic" i... |  5.0   |
    | Levana Safe N'See Digital ... | This is the first review I... |  1.0   |
    | Snuza Portable Baby Moveme... | I would have given the pro... |  1.0   |
    | Fisher-Price Ocean Wonders... | We have not had ANY luck w... |  2.0   |
    | VTech Communications Safe ... | This is my second video mo... |  1.0   |
    | Safety 1st High-Def Digita... | We bought this baby monito... |  1.0   |
    | Chicco Cortina KeyFit 30 T... | My wife and I have used th... |  1.0   |
    | Prince Lionheart Warmies W... | *****IMPORTANT UPDATE*****... |  1.0   |
    | Valco Baby Tri-mode Twin S... | I give one star to the dim... |  1.0   |
    | Adiri BPA Free Natural Nur... | I will try to write an obj... |  2.0   |
    | Munchkin Nursery Projector... | Updated January 3, 2014.  ... |  1.0   |
    | The First Years True Choic... | Note: we never installed b... |  1.0   |
    | Nuby Natural Touch Silicon... | I'm honestly confused by s... |  1.0   |
    | Peg-Perego Tatamia High Ch... | I ordered this high chair ... |  1.0   |
    |    Fisher-Price Royal Potty   | This was the worst potty e... |  1.0   |
    | Safety 1st Exchangeable Ti... | I thought it sounded great... |  1.0   |
    | Safety 1st Lift Lock and S... | Don't buy this product. If... |  1.0   |
    | Evenflo Take Me Too Premie... | I am absolutely disgusted ... |  1.0   |
    | Cloth Diaper Sprayer--styl... | I bought this sprayer out ... |  1.0   |
    | The First Years 3 Pack Bre... | I purchased several of the... |  1.0   |
    +-------------------------------+-------------------------------+--------+
    +-------------------------------+-----------+-------------------+
    |           word_count          | sentiment |       scores      |
    +-------------------------------+-----------+-------------------+
    | {'raining': 1L, 'all': 8L,... |     1     | 7.8041506821e-100 |
    | {'all': 2L, 'just': 1L, 'b... |     -1    | 6.83650885514e-25 |
    | {'contacted': 1L, 'being':... |     -1    | 2.12654510824e-24 |
    | {'fishstarfish': 1L, 'all'... |     -1    | 2.24582080779e-23 |
    | {'all': 4L, 'reviewers': 1... |     -1    | 1.32962966148e-22 |
    | {'all': 3L, 'being': 1L, '... |     -1    | 2.06872097469e-20 |
    | {'all': 4L, 'wrestle': 1L,... |     -1    | 5.93881994672e-20 |
    | {'less': 1L, 'move': 1L, '... |     -1    |  6.2851001654e-20 |
    | {'limited': 2L, 'forget': ... |     -1    | 8.05528712691e-20 |
    | {'all': 2L, 'forget': 1L, ... |     -1    | 8.46521724944e-20 |
    | {'son': 1L, 'all': 2L, 'se... |     -1    | 1.52853945171e-19 |
    | {'all': 3L, 'go': 1L, 'pow... |     -1    | 1.77901889379e-19 |
    | {'now': 1L, 'just': 3L, 'b... |     -1    | 1.15227353848e-18 |
    | {'just': 2L, 'food': 2L, '... |     -1    | 1.26175666136e-18 |
    | {'would': 1L, 'old': 1L, '... |     -1    | 1.60282966315e-18 |
    | {'just': 1L, 'whoever': 1L... |     -1    |  7.0488741171e-18 |
    | {'all': 2L, 'money': 1L, '... |     -1    | 9.84839237567e-18 |
    | {'all': 1L, 'managed': 1L,... |     -1    | 1.00120730395e-17 |
    | {'all': 1L, '45': 1L, 'jus... |     -1    |  1.169063556e-17  |
    | {'all': 1L, 'just': 2L, 'r... |     -1    | 1.22003532002e-17 |
    +-------------------------------+-----------+-------------------+
    [20 rows x 6 columns]
    
    

**Quiz Question**: Which of the following products are represented in the 20 most negative reviews?  [multiple choice]

## Compute accuracy of the classifier

We will now evaluate the accuracy of the trained classifer. Recall that the accuracy is given by


$$
\mbox{accuracy} = \frac{\mbox{# correctly classified examples}}{\mbox{# total examples}}
$$

This can be computed as follows:

* **Step 1:** Use the trained model to compute class predictions (**Hint:** Use the `predict` method)
* **Step 2:** Count the number of data points when the predicted class labels match the ground truth labels (called `true_labels` below).
* **Step 3:** Divide the total number of correct predictions by the total number of data points in the dataset.

Complete the function below to compute the classification accuracy:


```python
def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    ## YOUR CODE HERE
    # Class predictions according to GraphLab Create:
    model = model
    data['score'] = model.predict(data)
    
    # Compute the number of correctly classified examples
    ## YOUR CODE HERE
    correct  = (data['sentiment'] == data['score']).sum()

    # Then compute accuracy by dividing num_correct by total number of examples
    ## YOUR CODE HERE
    accuracy = (correct / len(data))
    
    return round(accuracy, 2)
```

Now, let's compute the classification accuracy of the **sentiment_model** on the **test_data**.


```python
get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
```




    0.91



**Quiz Question**: What is the accuracy of the **sentiment_model** on the **test_data**? Round your answer to 2 decimal places (e.g. 0.76).

**Quiz Question**: Does a higher accuracy value on the **training_data** always imply that the classifier is better?

## Learn another classifier with fewer words

There were a lot of words in the model we trained above. We will now train a simpler logistic regression model using only a subet of words that occur in the reviews. For this assignment, we selected a 20 words to work with. These are:


```python
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
```


```python
len(significant_words)
```




    20



For each review, we will use the **word_count** column and trim out all words that are **not** in the **significant_words** list above. We will use the [SArray dictionary trim by keys functionality]( https://dato.com/products/create/docs/generated/graphlab.SArray.dict_trim_by_keys.html). Note that we are performing this on both the training and test set.


```python
train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False)
```

Let's see what the first example of the dataset looks like:


```python
train_data[0]['review']
```




    'it came early and was not disappointed. i love planet wise bags and now my wipe holder. it keps my osocozy wipes moist and does not leak. highly recommend it.'



The **word_count** column had been working with before looks like the following:


```python
print train_data[0]['word_count']
```

    {'and': 3L, 'love': 1L, 'it': 3L, 'highly': 1L, 'osocozy': 1L, 'bags': 1L, 'leak': 1L, 'moist': 1L, 'does': 1L, 'recommend': 1L, 'was': 1L, 'wipes': 1L, 'disappointed': 1L, 'early': 1L, 'not': 2L, 'now': 1L, 'holder': 1L, 'wipe': 1L, 'keps': 1L, 'wise': 1L, 'i': 1L, 'planet': 1L, 'my': 2L, 'came': 1L}
    

Since we are only working with a subet of these words, the column **word_count_subset** is a subset of the above dictionary. In this example, only 2 `significant words` are present in this review.


```python
print train_data[0]['word_count_subset']
```

    {'love': 1L, 'disappointed': 1L}
    

## Train a logistic regression model on a subset of data

We will now build a classifier with **word_count_subset** as the feature and **sentiment** as the target. 


```python
simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target = 'sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None)
simple_model
```


<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 133416</pre>



<pre>Number of classes           : 2</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 20</pre>



<pre>Number of coefficients    : 21</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training-accuracy |</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>| 1         | 2        | 0.242171     | 0.862917          |</pre>



<pre>| 2         | 3        | 0.384273     | 0.865713          |</pre>



<pre>| 3         | 4        | 0.511362     | 0.866478          |</pre>



<pre>| 4         | 5        | 0.634450     | 0.866748          |</pre>



<pre>| 5         | 6        | 0.776552     | 0.866815          |</pre>



<pre>| 6         | 7        | 0.906644     | 0.866815          |</pre>



<pre>+-----------+----------+--------------+-------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>





    Class                         : LogisticClassifier
    
    Schema
    ------
    Number of coefficients        : 21
    Number of examples            : 133416
    Number of classes             : 2
    Number of feature columns     : 1
    Number of unpacked features   : 20
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.01
    
    Training Summary
    ----------------
    Solver                        : auto
    Solver iterations             : 6
    Solver status                 : SUCCESS: Optimal solution found.
    Training time (sec)           : 0.9487
    
    Settings
    --------
    Log-likelihood                : 44323.7254
    
    Highest Positive Coefficients
    -----------------------------
    word_count_subset[loves]      : 1.6773
    word_count_subset[perfect]    : 1.5145
    word_count_subset[love]       : 1.3654
    (intercept)                   : 1.2995
    word_count_subset[easy]       : 1.1937
    
    Lowest Negative Coefficients
    ----------------------------
    word_count_subset[disappointed]: -2.3551
    word_count_subset[return]     : -2.1173
    word_count_subset[waste]      : -2.0428
    word_count_subset[broke]      : -1.658
    word_count_subset[money]      : -0.8979



We can compute the classification accuracy using the `get_classification_accuracy` function you implemented earlier.


```python
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
```




    0.87



Now, we will inspect the weights (coefficients) of the **simple_model**:


```python
simple_model.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">class</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.2995449552</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0120888541331</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">disappointed</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.35509250061</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0504149888557</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">love</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.36543549368</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0303546295109</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">well</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.504256746398</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.021381300631</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">product</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.320555492996</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0154311321362</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">loves</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.67727145556</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0482328275384</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">little</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.520628636025</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0214691475665</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">work</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.621700012425</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0230330597946</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">easy</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.19366189833</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.029288869202</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">word_count_subset</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">great</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.94469126948</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0209509926591</td>
    </tr>
</table>
[21 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



Let's sort the coefficients (in descending order) by the **value** to obtain the coefficients with the most positive effect on the sentiment.


```python
simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21)
```

    +-------------------+--------------+-------+-----------------+-----------------+
    |        name       |    index     | class |      value      |      stderr     |
    +-------------------+--------------+-------+-----------------+-----------------+
    | word_count_subset |    loves     |   1   |  1.67727145556  | 0.0482328275384 |
    | word_count_subset |   perfect    |   1   |  1.51448626703  |  0.049861952294 |
    | word_count_subset |     love     |   1   |  1.36543549368  | 0.0303546295109 |
    |    (intercept)    |     None     |   1   |   1.2995449552  | 0.0120888541331 |
    | word_count_subset |     easy     |   1   |  1.19366189833  |  0.029288869202 |
    | word_count_subset |    great     |   1   |  0.94469126948  | 0.0209509926591 |
    | word_count_subset |    little    |   1   |  0.520628636025 | 0.0214691475665 |
    | word_count_subset |     well     |   1   |  0.504256746398 |  0.021381300631 |
    | word_count_subset |     able     |   1   |  0.191438302295 | 0.0337581955697 |
    | word_count_subset |     old      |   1   | 0.0853961886678 | 0.0200863423025 |
    | word_count_subset |     car      |   1   |  0.058834990068 | 0.0168291532091 |
    | word_count_subset |     less     |   1   | -0.209709815216 |  0.040505735954 |
    | word_count_subset |   product    |   1   | -0.320555492996 | 0.0154311321362 |
    | word_count_subset |    would     |   1   | -0.362308947711 | 0.0127544751985 |
    | word_count_subset |     even     |   1   |  -0.51173855127 | 0.0199612760261 |
    | word_count_subset |     work     |   1   | -0.621700012425 | 0.0230330597946 |
    | word_count_subset |    money     |   1   | -0.897884155776 | 0.0339936732836 |
    | word_count_subset |    broke     |   1   |  -1.65796447838 | 0.0580878907166 |
    | word_count_subset |    waste     |   1   |   -2.042773611  | 0.0644702932444 |
    | word_count_subset |    return    |   1   |  -2.11729659718 | 0.0578650807241 |
    | word_count_subset | disappointed |   1   |  -2.35509250061 | 0.0504149888557 |
    +-------------------+--------------+-------+-----------------+-----------------+
    [21 rows x 5 columns]
    
    

**Quiz Question**: Consider the coefficients of **simple_model**. There should be 21 of them, an intercept term + one for each word in **significant_words**. How many of the 20 coefficients (corresponding to the 20 **significant_words** and *excluding the intercept term*) are positive for the `simple_model`?


```python
tt1 = simple_model.coefficients.sort('value', ascending=False)
tt1.print_rows(num_rows=21)
```

    +-------------------+--------------+-------+-----------------+-----------------+
    |        name       |    index     | class |      value      |      stderr     |
    +-------------------+--------------+-------+-----------------+-----------------+
    | word_count_subset |    loves     |   1   |  1.67727145556  | 0.0482328275384 |
    | word_count_subset |   perfect    |   1   |  1.51448626703  |  0.049861952294 |
    | word_count_subset |     love     |   1   |  1.36543549368  | 0.0303546295109 |
    |    (intercept)    |     None     |   1   |   1.2995449552  | 0.0120888541331 |
    | word_count_subset |     easy     |   1   |  1.19366189833  |  0.029288869202 |
    | word_count_subset |    great     |   1   |  0.94469126948  | 0.0209509926591 |
    | word_count_subset |    little    |   1   |  0.520628636025 | 0.0214691475665 |
    | word_count_subset |     well     |   1   |  0.504256746398 |  0.021381300631 |
    | word_count_subset |     able     |   1   |  0.191438302295 | 0.0337581955697 |
    | word_count_subset |     old      |   1   | 0.0853961886678 | 0.0200863423025 |
    | word_count_subset |     car      |   1   |  0.058834990068 | 0.0168291532091 |
    | word_count_subset |     less     |   1   | -0.209709815216 |  0.040505735954 |
    | word_count_subset |   product    |   1   | -0.320555492996 | 0.0154311321362 |
    | word_count_subset |    would     |   1   | -0.362308947711 | 0.0127544751985 |
    | word_count_subset |     even     |   1   |  -0.51173855127 | 0.0199612760261 |
    | word_count_subset |     work     |   1   | -0.621700012425 | 0.0230330597946 |
    | word_count_subset |    money     |   1   | -0.897884155776 | 0.0339936732836 |
    | word_count_subset |    broke     |   1   |  -1.65796447838 | 0.0580878907166 |
    | word_count_subset |    waste     |   1   |   -2.042773611  | 0.0644702932444 |
    | word_count_subset |    return    |   1   |  -2.11729659718 | 0.0578650807241 |
    | word_count_subset | disappointed |   1   |  -2.35509250061 | 0.0504149888557 |
    +-------------------+--------------+-------+-----------------+-----------------+
    [21 rows x 5 columns]
    
    

**Quiz Question**: Are the positive words in the **simple_model** (let us call them `positive_significant_words`) also positive words in the **sentiment_model**?


```python
print len(tt1[tt1['value']>0 ]) # exclude intercept
```

    11
    

# Comparing models

We will now compare the accuracy of the **sentiment_model** and the **simple_model** using the `get_classification_accuracy` method you implemented above.

First, compute the classification accuracy of the **sentiment_model** on the **train_data**:


```python
get_classification_accuracy(sentiment_model, train_data, train_data['sentiment'])
```




    0.98



Now, compute the classification accuracy of the **simple_model** on the **train_data**:


```python
get_classification_accuracy(simple_model, train_data, train_data['sentiment'])
```




    0.87



**Quiz Question**: Which model (**sentiment_model** or **simple_model**) has higher accuracy on the TRAINING set?

Now, we will repeat this excercise on the **test_data**. Start by computing the classification accuracy of the **sentiment_model** on the **test_data**:


```python
get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
```




    0.91



Next, we will compute the classification accuracy of the **simple_model** on the **test_data**:


```python
get_classification_accuracy(simple_model, test_data, test_data['sentiment'])
```




    0.87



**Quiz Question**: Which model (**sentiment_model** or **simple_model**) has higher accuracy on the TEST set?

## Baseline: Majority class prediction

It is quite common to use the **majority class classifier** as the a baseline (or reference) model for comparison with your classifier model. The majority classifier model predicts the majority class for all data points. At the very least, you should healthily beat the majority class classifier, otherwise, the model is (usually) pointless.

What is the majority class in the **train_data**?


```python
num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print num_positive
print num_negative
```

    112164
    21252
    

Now compute the accuracy of the majority class classifier on **test_data**.

**Quiz Question**: Enter the accuracy of the majority class classifier model on the **test_data**. Round your answer to two decimal places (e.g. 0.76).


```python
round(num_positive/ (num_positive + num_negative), 2)
```




    0.84



**Quiz Question**: Is the **sentiment_model** definitely better than the majority class classifier (the baseline)?
