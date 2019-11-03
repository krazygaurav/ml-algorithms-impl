'''
@Author - Gaurav Singhal
@Description - Implementation of Linear regression using Gradient descent as optimization algorithm
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
            # x0 = 1
            instance = [1] + [float(x) for x in row.split(",")]
            X.append(instance[0:-1])
            y.append(instance[-1])
        f.close()
        weights = [float(0)]*len(X[0])
        return X, y, weights

def forwardPropogation(X, weights):
    y_predict = []
    for i in range(len(X)):
        y_cap = 0
        # equation of line: y_cap = w0x0 + w1x1 + w2x2 + .....
        for j in range(len(X[i])):
            y_cap += (X[i][j] * weights[j])
        y_predict.append(y_cap)
    return y_predict

def backPropogation(X, y, y_predict, weights, learningRate):
    e_sq, gradient = 0, [0]*len(weights)
    for i in range(len(y)):
        error = y[i] - y_predict[i]
        e_sq += error*error
        for j in range(len(gradient)):
            gradient[j] += error*X[i][j]
    for i in range(len(gradient)):
        weights[i] = weights[i] + learningRate*gradient[i]
    return weights, e_sq
        
def execute(filePath, learningRate, threshold):
    X, y, weights = loadDataset(filePath)
    if X is None:
        return
    error_sq_old, iteration = 100000000000000, 0
    outputFilePath = filePath + "-" + str(learningRate) + "-" + str(threshold) + ".csv";
    outputFile = open(outputFilePath, "w+")
    while(True):
        y_predict = forwardPropogation(X, weights)
        line = str(iteration) + "," + ",".join(["%.4f"%x for x in weights]) + ","
        # Updating weights and get squared error
        weights, error_sq = backPropogation(X, y, y_predict, weights, learningRate)
        if(error_sq_old - error_sq <= threshold or error_sq == float('+inf') or error_sq == float('-inf')):
            break
        line += "%.4f"%error_sq + "\n"
        outputFile.write(line)
        error_sq_old = error_sq
        iteration += 1
    outputFile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Leaarning task 1')
    parser.add_argument('--data', dest="data", help='Path to the dataset file)')
    parser.add_argument('--learningRate', dest="learningRate", help='Specify the learning rate (neta)')
    parser.add_argument('--threshold', dest="threshold", help='Specify the threshold for the mean square error')
    
    args = parser.parse_args()
    if(args.data == None or args.learningRate == None or args.threshold == None):
        print("\nusage: python3 <PROGRAM_NAME>.py --data xxx.csv --learningRate xx.xx --threshold xx.xx\n")
        exit(0)
    else:
        try:
            learningRate = float(args.learningRate)
            threshold = float(args.threshold)
            execute(args.data, learningRate, threshold)
        except:
            print("\nFloat value required for arguments --threshold and --learningRate\n")
