# 100 Days Of ML - Log

Rules: https://github.com/llSourcell/100_Days_of_ML_Code

## Day 7: July 23, 2018
**Today's Progress**:
- Went through [this tutorial](https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html) to to apply Natural Language Processing (tokenization, lemmatization, tf-idf) in NLTK and SciKit-Learn.
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
