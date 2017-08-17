//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// TestHelper.cs -- Help functions for CNTK Library C# model training tests.
//
using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    public class TestHelper
    {
        public enum Activation
        {
            None,
            ReLU,
            Sigmoid,
            Tanh
        }
        public static Function Dense(Variable input, int outputDim, DeviceDescriptor device, 
            Activation activation = Activation.None, string outputName = "")
        {
            if (input.Shape.Rank != 1)
            {
                // 
                int newDim = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDim });
            }

            Function fullyConnected = FullyConnectedLinearLayer(input, outputDim, device, outputName);
            switch (activation)
            {
                default:
                case Activation.None:
                    return fullyConnected;
                case Activation.ReLU:
                    return CNTKLib.ReLU(fullyConnected);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(fullyConnected);
                case Activation.Tanh:
                    return CNTKLib.Tanh(fullyConnected);
            }
        }

        public static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device,
            string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];

            int[] s = { outputDim, inputDim };
            var timesParam = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");

            int[] s2 = { outputDim };
            var plusParam = new Parameter((NDShape)s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, new Variable(timesFunction), outputName);
        }

        //Function Convolution(int[] filter_shape,     // shape of receptive field, e.g. (3,3)
        //    int num_filters, //  e.g. 64 or None (which means 1 channel and don't add a dimension)
        //    bool sequential = false, // time convolution if True (filter_shape[0] corresponds to dynamic axis)
        //    Activation activation = Activation.None,
        //    init = default_override_or(glorot_uniform()),
        //    bool pad= false,
        //    int strides= 1,
        //    bool bias= true,
        //    float init_bias= 0,
        //    int reduction_rank= 1, // (0 means input has no depth dimension, e.g. audio signal or B&W image)
        //    int max_temp_mem_size_in_samples= 0,
        //    string name= "")
        //{

        //}

        public static void SaveAndReloadModel(ref Function function, IList<Variable> variables, DeviceDescriptor device, uint rank = 0)
        {
            string tempModelPath = "feedForward.net" + rank;
            File.Delete(tempModelPath);

            IDictionary<string, Variable> inputVarUids = new Dictionary<string, Variable>();
            IDictionary<string, Variable> outputVarNames = new Dictionary<string, Variable>();

            foreach (var variable in variables)
            {
                if (variable.IsOutput)
                    outputVarNames.Add(variable.Owner.Name, variable);
                else
                    inputVarUids.Add(variable.Uid, variable);
            }

            function.Save(tempModelPath);
            function = Function.Load(tempModelPath, device);

            File.Delete(tempModelPath);

            var inputs = function.Inputs;
            foreach (var inputVarInfo in inputVarUids.ToList())
            {
                var newInputVar = inputs.First(v => v.Uid == inputVarInfo.Key);
                inputVarUids[inputVarInfo.Key] = newInputVar;
            }

            var outputs = function.Outputs;
            foreach (var outputVarInfo in outputVarNames.ToList())
            {
                var newOutputVar = outputs.First(v => v.Owner.Name == outputVarInfo.Key);
                outputVarNames[outputVarInfo.Key] = newOutputVar;
            }
        }

        public static Value MinibatchDataToValue(MinibatchData minibatchData, int[] imageDims, int minibatchSize)
        {
            int imageSize = imageDims.Aggregate((d1, d2) => d1 * d2);
            int sequenceLength = 1;
            NDShape shape = NDShape.CreateNDShape(new int[] { imageSize, sequenceLength, minibatchSize });
            Variable inputVariableForShape = Variable.InputVariable(shape, DataType.Float);

            int[] imageBatchDim = new int[imageDims.Length + 2];
            imageDims.CopyTo(imageBatchDim, 0);
            imageBatchDim[imageBatchDim.Length - 2] = sequenceLength;
            imageBatchDim[imageBatchDim.Length - 1] = minibatchSize;

            System.Collections.Generic.IList<System.Collections.Generic.IList<float>> bufs =
                minibatchData.data.GetDenseData<float>(inputVariableForShape);
            // NDShape imageBatchShape = NDShape.CreateNDShape(imageBatchDim);
            Value imageBatchValue = Value.CreateBatch<float>(imageDims, bufs[0], minibatchData.data.Device);
            return imageBatchValue;
        }


        public static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                double trainLossValue = trainer.PreviousMinibatchLossAverage();
                double evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                Console.WriteLine($"Minibatch: {minibatchIdx} CrossEntropy loss = {trainLossValue}, Evaluation criterion = {evaluationValue}");
            }
        }

        public static void PrintOutputDims(Function function, string functionName)
        {
            NDShape shape = function.Output.Shape;

            if (shape.Rank == 3)
            {
                Console.WriteLine($"{functionName} dim0: {shape[0]}, dim1: {shape[1]}, dim2: {shape[2]}");
            }
            else
            {
                Console.WriteLine($"{functionName} dim0: {shape[0]}");
            }
        }

    }
}
