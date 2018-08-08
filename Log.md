# 100 Days Of ML - Log

Rules: https://github.com/llSourcell/100_Days_of_ML_Code

## Day 17: August 7, 2018
**Today's Progress**: Started going through the [Community Starter Kit](https://lab.github.com/courses/community-starter-kit) from [GitHub courses](https://lab.github.com/courses).

**Discussion:** Although a bit tangential to Machine Learning, I think it's important for me to have a more in-depth understanding of GitHub--especially about issues, pull requests, community contributions, etc. This will allow me to contribute to open-source Machine Learning libraries or start my own in the future.

## Day 16: August 6, 2018
**Today's Progress**: Similarly to Day 14, Watched a recording of another NLP presentation from the same Microsoft ML conference.

## Day 15: August 5, 2018
**Today's Progress**:
- Continued reading the book by A. Géron
  - Chapter 4 (Training Models) sections "Regularized Linear Models" (Ridge Regression, Lasso Regression)
  - Finished reading Chapter 9 (Up and Running with TensorFlow)
  - Started building an MNIST logistic regression classifier using TensorFlow in a Colab notebook. I'm not following any tutorial but rather just trying to use the code snippets from this chapter in the book, or the TensorFlow API documentation. I managed to get the training to work, but now I just need to find out how to compute the accuracy of the trained model on the test set.

## Day 14: August 2, 2018
**Today's Progress**: Watched a recording of an NLP presentation from an internal Microsoft ML conference.

## Day 13: July 30, 2018
**Today's Progress**: Trying to learn a bit more about the bias-variance tradeoff.
- Book "Pattern Recognition and Machine Learning" by C. Bishop. Section 3.2 "The Bias-Variance Decomposition"
- https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec04.pdf
- https://elitedatascience.com/bias-variance-tradeoff
- http://www.r2d3.us/visual-intro-to-machine-learning-part-2/
- Played a bit with a jupyter notebook doing some polynomial regressions with data of small sample size to experiment with overfitting

**Thoughts**:
- I still haven't found a proof that bias necessarily needs to go down when variance goes up and vice versa given a certain problem with a given sample size. Am I missing something?
- The bias-variance decomposition seems to be framed as a regression problem with sum of squares loss. Is there such a decomposition for classification problems? What about for problems that don't have a square loss?

## Day 12: July 29, 2018
**Today's Progress**:
- Continued with A. Géron's book. Multilabel Classification, Multioutput Classification. Started Chaper 4 ("Training Models"). Linear Regression: The Normal Equation, Computational Complexity. Jumped over the sections on Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent, because I had already gone through them (on day 6?). Continued with Polinomial Regression, Learning Curves, The Bias/Variance Tradeoff. 
- Read about "[Bias–variance decomposition of squared error](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff#Bias%E2%80%93variance_decomposition_of_squared_error)" on Wikipedia. Instead of the usual hand-wavy description of the bias-variance tradeoff, this section actually shows a decomposition in mathematical terms of expected square error into bias, variance, and a randomness term.

**Thoughts**:
- It's the first time I see learning curves being plotted as a function of training set size (which requires retraining from scratch for size 1, then for size 2, etc.). Before, I had only known about learning curves as a function of training iteration.
- It's great to learn about the actual Bias-Variance decomposition in rigorous terms. However, I don't think that this decomposition really shows that there is a tradeoff or tension between these two -- as far as I know, there's nothing that constrains the equation such that one *has* to go down when the other goes up. I need to better understand if there's a rigorous proof in terms of this decomposition actually causing a tradeoff to occur.

## Day 11: July 28, 2018
**Today's Progress**:
- Continued reading a little bit of the book by A. Géron. Read the part about evaluation of binary or multiclass classifiers with Precision/Recall, ROC curves, and ROC area under curve. Also learned that when doing multi-class classification with binary classifiers, Scikit-Learn uses a one-vs-the-rest approach in most cases, but it uses a one-vs-one approach for SVMs for efficiency.
- Finally conquered my fear of submitting a script to Kaggle. As my "Hello World", I made a submission to the [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) contest. I wrote the code directly in a Kaggle kernel (Jupyter notebook style) and submitted the solution produced by that notebook. Since for me this was simply a "Hello World", I used a super simple linear SGDClassifier classifier with no customizations or preprocessing, which scored a really low 85.8% accuracy. This put me at position 2483 out of 2645 in the contest 😅.

## Day 10: July 26, 2018
**Today's Progress**: Continued reading the book by A. Géron.
- Read about Modularity, and Sharing Variables in TensorFlow.
- Then flipped back to the first half of the book to learn about applying and evaluating classification algorithms in Scikit-Learn.

## Day 9: July 25, 2018
**Today's Progress**: Created a PowerPoint and video presentation of applying the findings from the previous two days to a custom dataset. This was for the Microsoft Hackathon.

## Day 8: July 24, 2018
**Today's Progress**:
- Continued working on the tutorial from yesterday
- Experimented with clustering of the tf-idf vectors that I had created for the tutorial

## Day 7: July 23, 2018
**Today's Progress**:
- Went through [this tutorial](https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html) to apply Natural Language Processing (tokenization, lemmatization, tf-idf) in NLTK and SciKit-Learn.
- Installed a dark theme for Jupyter Notebook 😎 ([Jupyter Themes](https://github.com/dunovank/jupyter-themes)).

## Day 6: July 22, 2018
**Today's Progress**:
- Continued learning about TensorFlow in Geron's book (Linear Regression with TensorFlow, Implementing Gradient Descent, Feeding Data to the Training Algorithm, Saving and Restoring Models, Visualizing the Graph and Training Curves Using TensorBoard, Name Scopes)
- Reviewed the different types of computational differentiation in the appendix of the same book:
  - Manual Differentiation: deriving the formula for the derivative by hand and explicitly coding it
  - Symbolic Differentiation: based on a computation graph, follow rules to create a new graph that computes the derivative
  - Numerical Differentiation: compute ((f(x, y) - f(x + epsilon, y)) / epsilon) for some small epsilon (note: the answer can be very inexact depending on the function and on the epsilon)
  - Forward-Mode Autodiff: compute f(x+epsilon, y) using algebra on [dual numbers](https://en.wikipedia.org/wiki/Dual_number). Uses the property that f(x+epsilon, y) = d(x, y) + d/dx f(x, y) * epsilon, where epsilon is defined as a number with the properties that epsilon != 0 and epsilon*epsilon=0.
  - Reverse-Mode Autodiff: forward and backward pass on the original computation graph. TensorFlow uses reverse-mode autodiff.

## Day 5: July 21, 2018
**Today's Progress**:
- Started reading the section about Tensorflow in Geron's book.

**Thoughts**: TensorFlow is still confusing for me. Conceptually, I understand Tensorflow (I know what purpose it serves). But I still don't fully understand the mechanics of its API, which is more "low-level" than, say, Keras. I'm not sure of why we need the concept of Sessions. I'm also not sure why there is a need for Graph objects; can't graphs be inferred based on connectivity?


## Day 4: July 19, 2018
**Today's Progress**:

- I read through the section "4.1. Pipeline and FeatureUnion: combining estimators" in the Scikit-Learn documentation http://scikit-learn.org/stable/modules/pipeline.html
- Found this cool [Scikit-Learn Cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf) by DataCamp and started going through it. I'm still pending going through the part titled "Evaluate Your Model’s Performance" and the one titled "Tune Your Model".

**Thoughts**: I think that Scikit-learn has finally "clicked" for me today. I like how everything has a standardized interface (fit(), transform(), predict(), etc.) and how you can build entire data pipelines combining different preprocessing and training steps.

## Day 3: July 18, 2018
**Today's Progress**:

Explored a bit through the SciKit Learn documentation; ended up reading some of the [APIs for Generalized Linear Models](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model). I still haven't read through the more human-readable [guide about Generalized Linear Models](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression).

Tried out Google CoLab Jupyter Notebooks (https://colab.research.google.com/). Ran a few of the linear models mentioned above with some synthetic data. Practiced using `??` and `?` in the Jupyter notebook to access the documentation.

**Thoughts**:
- Scikit-learn has a lot of different kinds of linear model variants available. Many more than I had expected.
- CoLab is like Jupyter but with a "Material Design" visual style. The design looks pretty clean, and I was pleasantly surprised that they have a "Search in Stack Overflow" button when I get a Python error.


## Day 2: July 17, 2018
 
**Today's Progress**: Finished reading Chapter 2 of the book. Grid search; feature importance; evaluating on the test set; launching, monitoring and maintaining the system.

**Thoughts**:

I especially liked the section "Launch, Monitor, and Maintain Your System". It explains about the process of having the pipeline running in practice, making sure to monitor it for performance degradation or corrupted data and re-train it every once in a while if the data distribution drifts. It also suggests having the system's performance evaluated by an expert or by mechanical turkers ("You need to plug the human evaluation pipeline into your system").

Quote from the chapter's conclusion: "it is probably preferable to be comfortable with the overall process and know three or four algorithms well rather than to spend all your time exploring advanced algorithms and not enough time on the overall process"


## Day 1: July 16, 2018
 
**Today's Progress**:

Continuing to read through "Hands-On Machine Learning with Skikit-Learn & Tensorflow" 1st Edition by Aurélien Géron. Going through Chapter 2 (End-to-End Machine Learning Project) pages 66-72, learning about custom transformers, feature scaling, transformation pipelines, training and evaluating models, and using cross-validation. All of these in Scikit-Learn.

**Thoughts**: I'm currently focusing on Chapter 2 (End-to-End Machine Learning Project) to make sure I have my foundations in terms of data preparation and Scikit-Learn.


## Day 0: July 15, 2018
 
**Today's Progress**: Set up this git repo after I learned about this challenge.

**Thoughts**: Seems like a cool challenge. I especially hope this challenge will allow me to practice the hands-on side of Machine Learning (my past focus has been a lot on theory and algorithms).
