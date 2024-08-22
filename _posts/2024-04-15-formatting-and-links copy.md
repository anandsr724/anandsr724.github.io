---
layout: post
title: Solving Handwritten Equations with Deep Learning - A Step-by-Step Guide
date: 2024-04-15 16:40:16
description: this is a test
tags: formatting links
categories: sample-posts
---

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

#### Hipster list

- brunch
- fixie
- raybans
- messenger bag


Hoodie Thundercats retro, tote bag 8-bit Godard craft beer gastropub. Truffaut Tumblr taxidermy, raw denim Kickstarter sartorial dreamcatcher. Quinoa chambray slow-carb salvia readymade, bicycle rights 90's yr typewriter selfies letterpress cardigan vegan.

<hr>

Pug heirloom High Life vinyl swag, single-origin coffee four dollar toast taxidermy reprehenderit fap distillery master cleanse locavore. Est anim sapiente leggings Brooklyn ea. Thundercats locavore excepteur veniam eiusmod. Raw denim Truffaut Schlitz, migas sapiente Portland VHS twee Bushwick Marfa typewriter retro id keytar.

> We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
> —Anais Nin

Fap aliqua qui, scenester pug Echo Park polaroid irony shabby chic ex cardigan church-key Odd Future accusamus. Blog stumptown sartorial squid, gastropub duis aesthetic Truffaut vero. Pinterest tilde twee, odio mumblecore jean shorts lumbersexual.
