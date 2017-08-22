from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

sc = MLUtils.loadLibSVMFile(path="data/mllib/sample_svm_data.txt")

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


data = sc.textFile("data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

print(parsedData)
# # Build the model
# model = SVMWithSGD.train(parsedData, iterations=100)
#
# # Evaluating the model on training data
# labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
# trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
# print("Training Error = " + str(trainErr))
#
# # Save and load model
# model.save(sc, "myModelPath")
# sameModel = SVMModel.load(sc, "myModelPath")