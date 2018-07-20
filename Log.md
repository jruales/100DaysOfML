# 100 Days Of ML - Log

Rules: https://github.com/llSourcell/100_Days_of_ML_Code

## Day 4: July 19, 2018
**Today's Progress**:

- I read through the section "4.1. Pipeline and FeatureUnion: combining estimators" in the Scikit-Learn documentation http://scikit-learn.org/stable/modules/pipeline.html
- Found this cool [Scikit-Learn Cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf) by DataCamp and went through it. I'm just missing to go through the part titled "Evaluate Your Model’s Performance" and the one titled "Tune Your Model".

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
