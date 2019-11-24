'''
@Author - Gaurav Singhal
@Description - Implementation of Perceptron algorithm with Delta rule (Batch mode)
'''

import argparse

# Load dataset
def loadDataset(filePath):
    X, y = [], []
    try:
        f =open(filePath, "r")
    except:
        print("\nError Opening file\n")
        return None, None, None
    else:
        for row in f.readlines():
            instance = row.strip().split()

            # x0 = 1
            if instance[0] == 'A':
                y.append(1)
            elif instance[0] == 'B':
                y.append(0)
            X.append([1] + [float(x) for x in instance[1:]])
        f.close()
        weights = [float(0)]*len(X[0])
        return X, y, weights

def activation(val):
    return 1 if val>0 else 0
    
def forwardPropogation(X, weights):
    y_predict = []
    for i in range(len(X)):
        y_cap = 0
        # equation of line: y_cap = w0x0 + w1x1 + w2x2 + .....
        for j in range(len(X[i])):
            y_cap += (X[i][j] * weights[j])
        y_predict.append(activation(y_cap))
    return y_predict

def backPropogation(X, y, y_predict, weights, learningRate):
    e_sq, gradient = 0, [0]*len(weights)
    for i in range(len(y)):
        error = y[i] - y_predict[i]
        e_sq += abs(error)
        for j in range(len(gradient)):
            gradient[j] += error*X[i][j]
    for i in range(len(gradient)):
        weights[i] = weights[i] + learningRate*gradient[i]
    return weights, e_sq
    
def runEpochs(X, y, weights, learningRate, annealing):
    iteration = 0
    line = ""
    while(True):
        iteration += 1
        y_predict = forwardPropogation(X, weights)
        weights, error_sq = backPropogation(X, y, y_predict, weights, learningRate if annealing == False else learningRate/iteration)
        line += "%d"%error_sq + "\t"
        if(iteration == 101):
            break
    return line
    
    
def execute(filePath, outputPath):
    X, y, weights1 = loadDataset(filePath)
    weights2 = weights1.copy()
    if X is None:
        return
    # Set default as per question
    learningRate = 1
    withoutAnnealing = runEpochs(X, y, weights1, learningRate, False)
    withAnnealing = runEpochs(X, y, weights2, learningRate, True)
    
    outputFile1 = open(outputPath + "/output.tsv", "w+")
    outputFile1.write(withoutAnnealing + "\n")
    outputFile1.write(withAnnealing)
    
    outputFile1.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Leaarning task 1')
    parser.add_argument('--data', dest="data", help='Path to the dataset file)')
    parser.add_argument('--output', dest="output", help='Path where to save output file')
    
    args = parser.parse_args()
    if(args.data == None or args.output == None):
        print("\nusage: python3 <PROGRAM_NAME>.py --data xxx.tsv --output xxxx\n")
        exit(0)
    else:
            execute(args.data,  args.output)