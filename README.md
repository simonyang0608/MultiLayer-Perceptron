# Multiple-Layer Perceptron
Using C# to implement multiple layer perceptron simulator. Learning algorithm for more details can refer to [this](https://en.wikipedia.org/wiki/Perceptron#Learning_algorithm) Wikipedia page.

The dataset consists of multi-dimensional input sample set and ground truth set, The sample set will be given to training function to predict the ground truth set.

The purpose of program is training to tune a number of groups of shifting biases and synaptic(scaling) weights, and display the results in the graphical interface.

## FeedForward Pass
![擷取](https://user-images.githubusercontent.com/31026907/218099407-7b633069-d07e-4d55-a40e-3cfbf041f898.PNG)


## BackPropagation Pass
![擷取](https://user-images.githubusercontent.com/31026907/218099875-c1987303-799b-4921-a976-81ae5f2a3eb6.PNG)


# Setup & Run Code

## Getting Started
- Clone this repo to your local

```
git clone https://github.com/simonyang0608/MultiLayer-Perceptron
cd MultiLayer-Perceptron
```

![擷取](https://user-images.githubusercontent.com/31026907/218102800-2a904e69-1d40-4402-8232-66b0eb9c71ef.PNG)

1. Input random sample set
2. Adjustable parameters
3. Start training button
4. Epochs
5. Training loss(curve) visualization
6. Learned hyperline visualization
7. Weights and biases distribution

## Demo
![20230210_214135](https://user-images.githubusercontent.com/31026907/218107731-bb23dd17-0435-4c62-8688-4c06e4bf7af7.gif)

# Conclusion

## Large size sample set will tend to converge faster than the small size, with the same settings of hidden layer and learning rate.

- Small size sample set

![20230211_114623](https://user-images.githubusercontent.com/31026907/218237420-544cfd2a-5ee1-4216-a4ba-528b525212fc.gif)

- Large size sample set

![20230211_115916](https://user-images.githubusercontent.com/31026907/218238123-727738b7-1c90-4c24-84bf-2a58284738cd.gif)

## Different size (below 100) of hidden layer with suitable learning rate will tend to get good result of convergence (no matter the size of sample set).

- Small size (below 20) hidden layer with the learning rate ranging from 0.08~0.007 will tend to get good result.

![20230210_220908](https://user-images.githubusercontent.com/31026907/218112653-ce69aee4-4a7c-4d63-9a15-2427b3a62d07.gif)

- Large size (from 20 to 100) hidden layer with the learning rate ranging from 0.03~0.006 will tend to get good result.

![20230210_224842](https://user-images.githubusercontent.com/31026907/218121816-4198f204-804a-4914-a638-fb3a75c00535.gif)

## Larger size hidden layer (over 100) is not recommended since the simple linear regression and prediction task no need to use such complicated network, it may lead to over-fitting, results in low converge speed.

# Contributing
Please feel free to use it if you are interested in fixing issues and contributing directly to the code base.

# License
Multilayer-Perceptron is released under the MIT license. See the [LICENSE](https://github.com/simonhandsome/MultiLayer-Perceptron/blob/main/LICENSE) file for details.
