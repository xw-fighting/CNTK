using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    public class CifarResNetTest
    {
        static Function ConvBatchNormalizationReLULayer(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, int hStride, int vStride, 
            double wScale, double bValue, double scValue, int bnTimeConst, bool spatial, DeviceDescriptor device)
        {
            var convBNFunction = ConvBatchNormalizationLayer(input, outFeatureMapCount, kernelWidth, kernelHeight, hStride, vStride, wScale, bValue, scValue, bnTimeConst, spatial, device);
            return CNTKLib.ReLU(convBNFunction);
        }

        static Function ConvBatchNormalizationLayer(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, int hStride, int vStride, 
            double wScale, double bValue, double scValue, int bnTimeConst, bool spatial, DeviceDescriptor device)
        {
            int numInputChannels = input.Shape[input.Shape.Rank - 1];

            var convParams = new Parameter(new int[]{ kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount }, 
                DataType.Float, CNTKLib.GlorotUniformInitializer(wScale, -1, 2), device);
            var convFunction = CNTKLib.Convolution(convParams, input, new int[]{ hStride, vStride, numInputChannels });

            var biasParams = new Parameter(new int[]{ NDShape.InferredDimension }, (float)bValue, device, "");
            var scaleParams = new Parameter(new int[]{ NDShape.InferredDimension }, (float)scValue, device, "");
            var runningMean = new Constant(new int[]{ NDShape.InferredDimension }, 0.0f, device);
            var runningInvStd = new Constant(new int[]{ NDShape.InferredDimension }, 0.0f, device);
            var runningCount = Constant.Scalar(0.0f, device);
            return CNTKLib.BatchNormalization(convFunction, scaleParams, biasParams, runningMean, runningInvStd, runningCount, 
                spatial, (double)bnTimeConst, 0.0, 1e-5 /* epsilon */);
        }

        static Function ResNetNode(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, double wScale, double bValue,
            double scValue, int bnTimeConst, bool spatial, DeviceDescriptor device)
        {
            var c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
            var c2 = ConvBatchNormalizationLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
            var p = CNTKLib.Plus(c2, input);
            return CNTKLib.ReLU(p);
        }

        static Function ResNetNodeInc(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, double wScale, double bValue, 
            double scValue, int bnTimeConst, bool spatial, Variable wProj, DeviceDescriptor device)
        {
            var c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
            var c2 = ConvBatchNormalizationLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

            var cProj = ProjectLayer(wProj, input, 2, 2, bValue, scValue, bnTimeConst, device);

            var p = CNTKLib.Plus(c2, cProj);
            return CNTKLib.ReLU(p);
        }

        static Function ProjectLayer(Variable wProj, Variable input, int hStride, int vStride, double bValue, double scValue, int bnTimeConst, 
            DeviceDescriptor device)
        {
            int outFeatureMapCount = wProj.Shape[0];
            var b = new Parameter(new int[]{ outFeatureMapCount }, (float)bValue, device, "");
            var sc = new Parameter(new int[]{ outFeatureMapCount }, (float)scValue, device, "");
            var m = new Constant(new int[]{ outFeatureMapCount }, 0.0f, device);
            var v = new Constant(new int[]{ outFeatureMapCount }, 0.0f, device);

            var n = Constant.Scalar(0.0f, device);

            int numInputChannels = input.Shape[input.Shape.Rank - 1];

            var c = CNTKLib.Convolution(wProj, input, new int[]{ hStride, vStride, numInputChannels }, new bool[]{ true }, new bool[] { false });
            return CNTKLib.BatchNormalization(c, sc, b, m, v, n, true /*spatial*/, (double)bnTimeConst, 0, 1e-5, false); 
        }

        static Constant GetProjectionMap(int outputDim, int inputDim, DeviceDescriptor device)
        {
            if (inputDim > outputDim)
                throw new Exception("Can only project from lower to higher dimensionality");

            float[] projectionMapValues = new float[inputDim * outputDim];
            for (int i = 0; i < inputDim * outputDim; i++)
                projectionMapValues[i] = 0;
            for (int i = 0; i<inputDim; ++i)
                projectionMapValues[(i * (int)inputDim) + i] = 1.0f;

            var projectionMap = new NDArrayView(DataType.Float, new int[]{ 1, 1, inputDim, outputDim }, device);
            projectionMap.CopyFrom(new NDArrayView(new int[]{ 1, 1, inputDim, outputDim }, projectionMapValues, (uint)projectionMapValues.Count(), device));

            return new Constant(projectionMap);
        }

        static Function ResNetClassifier(Variable input, int numOutputClasses, DeviceDescriptor device, string outputName)
        {
            double convWScale = 7.07;
            double convBValue = 0;

            double fc1WScale = 0.4;
            double fc1BValue = 0;

            double scValue = 1;
            int bnTimeConst = 4096;

            int kernelWidth = 3;
            int kernelHeight = 3;

            double conv1WScale = 0.26;
            int cMap1 = 16;
            var conv1 = ConvBatchNormalizationReLULayer(input, cMap1, kernelWidth, kernelHeight, 1, 1, conv1WScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

            var rn1_1 = ResNetNode(new Variable(conv1), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn1_2 = ResNetNode(new Variable(rn1_1), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);
            var rn1_3 = ResNetNode(new Variable(rn1_2), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

            int cMap2 = 32;
            var rn2_1_wProj = GetProjectionMap(cMap2, cMap1, device);
            var rn2_1 = ResNetNodeInc(new Variable(rn1_3), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn2_1_wProj, device);
            var rn2_2 = ResNetNode(new Variable(rn2_1), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn2_3 = ResNetNode(new Variable(rn2_2), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

            int cMap3 = 64;
            var rn3_1_wProj = GetProjectionMap(cMap3, cMap2, device);
            var rn3_1 = ResNetNodeInc(new Variable(rn2_3), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn3_1_wProj, device);
            var rn3_2 = ResNetNode(new Variable(rn3_1), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn3_3 = ResNetNode(new Variable(rn3_2), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

            // Global average pooling
            int poolW = 8;
            int poolH = 8;
            int poolhStride = 1;
            int poolvStride = 1;
            var pool = CNTKLib.Pooling(new Variable(rn3_3), PoolingType.Average, 
                new int[] { poolW, poolH, 1 }, new int[] { poolhStride, poolvStride, 1 });

            TestHelper.PrintOutputDims(pool, "pool");
            
            // Output DNN layer
            var outTimesParams = new Parameter(new int[]{ numOutputClasses, 1, 1, cMap3 }, DataType.Float, 
                CNTKLib.GlorotUniformInitializer(fc1WScale, 1, 0), device);
            var outBiasParams = new Parameter(new int[] { numOutputClasses }, (float)fc1BValue, device, "");

            return CNTKLib.Plus(new Variable(CNTKLib.Times(outTimesParams, new Variable(pool))), outBiasParams, outputName);
        }

        static MinibatchSource CreateCifarMinibatchSource(ulong epochSize)
        {
            int imageHeight = 32;
            int imageWidth = 32;
            int numChannels = 3;
            uint numClasses = 10;
            var mapFilePath = "train_map.txt";
            var meanFilePath = "CIFAR-10_mean.xml";

            List<CNTKDictionary> transforms = new List<CNTKDictionary>{
                CNTKLib.ReaderCrop("RandomSide",
                    new Tuple<int, int>(0, 0),
                    new Tuple<float, float>(0.8f, 1.0f),
                    new Tuple<float, float>(0.0f, 0.0f),
                    new Tuple<float, float>(1.0f, 1.0f),
                    "uniRatio"),
                CNTKLib.ReaderScale(imageWidth, imageHeight, numChannels),
                CNTKLib.ReaderMean(meanFilePath)
            };

            var deserializerConfiguration = CNTKLib.ImageDeserializer(mapFilePath,
                "labels", numClasses,
                "features",
                transforms);

            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfiguration });

            // TODO: shall batch size be int or long?
            config.maxSamples = (uint)epochSize;

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }

        public static void TrainResNetCifarClassifier(DeviceDescriptor device, bool testSaveAndReLoad)
        {
            var minibatchSource = CreateCifarMinibatchSource(MinibatchSource.InfinitelyRepeat);
            var imageStreamInfo = minibatchSource.StreamInfo("features");
            var labelStreamInfo = minibatchSource.StreamInfo("labels");

            var inputImageShape = imageStreamInfo.m_sampleLayout;
            int numOutputClasses = (int)(labelStreamInfo.m_sampleLayout[0]);

            var imageInputName = "Images";
            var imageInput = CNTKLib.InputVariable(inputImageShape, imageStreamInfo.m_elementType, imageInputName);
            var classifierOutput = ResNetClassifier(imageInput, numOutputClasses, device, "classifierOutput");
            TestHelper.PrintOutputDims(classifierOutput, "classifierOutput");

            var labelsInputName = "Labels";
            var labelsVar = CNTKLib.InputVariable(new int[] { numOutputClasses }, labelStreamInfo.m_elementType, labelsInputName);
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labelsVar, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labelsVar, 5, "predictionError");

            if (testSaveAndReLoad)
            {
                Variable classifierOutputVar = new Variable(classifierOutput);
                Variable trainingLossVar = new Variable(trainingLoss);
                Variable predictionVar = new Variable(prediction);
                var imageClassifier = Function.Combine(new List<Variable>() { trainingLossVar, predictionVar, classifierOutputVar }, "ImageClassifier");
                TestHelper.SaveAndReloadModel(ref imageClassifier, new List<Variable> { imageInput, labelsVar, trainingLossVar, predictionVar, classifierOutputVar }, device);

                // Make sure that the names of the input variables were properly restored
                if ((imageInput.Name != imageInputName) || (labelsVar.Name != labelsInputName))
                    throw new Exception("One or more input variable names were not properly restored after save and load");

                trainingLoss = trainingLossVar;
                prediction = predictionVar.ToFunction();
                classifierOutput = classifierOutputVar.ToFunction();
            }


            TrainingParameterPerSampleScheduleDouble learningRatePerSample = new TrainingParameterPerSampleScheduleDouble(0.0078125);
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction,
                new List<Learner> { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) });

            uint minibatchSize = 32;
            int numMinibatchesToTrain = 2000;
            int outputFrequencyInMinibatches = 20;
            for (int i = 0; i < numMinibatchesToTrain; ++i)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                trainer.TrainMinibatch(new Dictionary<Variable, MinibatchData>()
                    { { imageInput, minibatchData[imageStreamInfo] }, { labelsVar, minibatchData[labelStreamInfo] } }, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            }
        }
    }
}
