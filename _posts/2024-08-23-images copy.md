---
layout: post
title: worekd Solving Handwritten Equations with Deep Learning
date: 2024-08-23 21:01:00
description: this is the copy
tags: formatting images
categories: sample-posts
thumbnail: assets/img/9.jpg
---

This is an example post with image galleries.
For the dataset, I used a publicly available dataset on Kaggle. You can find the dataset [here](https://www.pinterest.com) .

{% highlight python %}
classes_ideal = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']',
'cos', 'e', 'forward_slash', 'log', 'pi', 'sin', 'sqrt', 'tan', 'times']
{% endhighlight %}

In today's digital age, machine learning (ML) has revolutionized numerous fields, including image recognition and natural language processing. One intriguing problem that combines both these areas is the automatic solving of handwritten mathematical equations. Imagine taking a picture of a complex equation and having a machine instantly provide the solution. This blog post delves into how I built a machine learning model to achieve just that.

#### Problem Statement
The task at hand is to develop a system that can take an image of a handwritten mathematical equation and output the correct answer. 

### To tackle this, the problem can be broken down into two primary components:

- Reading and detecting the symbols: The system must recognize individual numbers and mathematical symbols from the image.
- Solving the equation: Once the symbols are identified, the system must correctly interpret and solve the equation.

### Symbol Detection with Convolutional Neural Networks (CNNs)

The first step is to accurately recognize the symbols in the image. We'll use a Convolutional Neural Network (CNN) for this purpose, as CNNs are particularly effective for image classification tasks.

## Code Blocks

**Target Classes:** We consider the following symbols and functions for our model:

{% highlight python %}
classes_ideal = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']',
'cos', 'e', 'forward_slash', 'log', 'pi', 'sin', 'sqrt', 'tan', 'times']
{% endhighlight %}

These symbols cover a broad range of arithmetic operations and functions commonly used in mathematical equations.

### Dataset
For the dataset, I used a publicly available dataset on Kaggle. You can find the dataset [here](https://www.pinterest.com) .

Analyzing the dataset, we noticed an imbalance in the number of images per class. For instance, some classes like ( and ) had only 700 images, while others like 1, 2, and 5 had a significantly larger number of images.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/8.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>




<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

Images can be made zoomable.
Simply add `data-zoomable` to `<img>` tags that you want to make zoomable.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/8.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

The rest of the images in this post are all zoomable, arranged into different mini-galleries.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/12.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/7.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
