# Multiple-Layer Perceptron
Using C# to implement multiple layer perceptron simulator.

The dataset consists of multi-dimensional input sample set and ground truth set, The sample set will be given to training function to predict the ground truth set.

The purpose of program is training to cut a number of groups of shifting biases and synaptic(scaling) weights, and display the results in the graphical interface.

## FeedForward Pass
![擷取](https://user-images.githubusercontent.com/31026907/218099407-7b633069-d07e-4d55-a40e-3cfbf041f898.PNG)


## BackPropagation Pass
![擷取](https://user-images.githubusercontent.com/31026907/218099875-c1987303-799b-4921-a976-81ae5f2a3eb6.PNG)


# Setup & run code

## Getting Started
- Clone this repo to your local

```
git clone https://github.com/simonhandsome/MultiLayer-Perceptron
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
### Different size of hidden layer with suitable learning rate will tend to get good result of convergence (no matter the size of sample set)

- Small size (below 20) of hidden layer with the learning rate ranging from 0.08~0.007 will tend to get good result

![20230210_220908](https://user-images.githubusercontent.com/31026907/218112653-ce69aee4-4a7c-4d63-9a15-2427b3a62d07.gif)


- Large size (up to 20~100 or more) of hidden layer with the learning rate ranging from 0.02~0.0001 will tend to get good result
