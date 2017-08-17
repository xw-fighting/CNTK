//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibShim.cs -- General C# Api methods
//
using System;
using System.Collections.Generic;

namespace CNTK
{
    public partial class CNTKLib
    {
        public static CNTKDictionary ReaderCrop(string cropType, Tuple<int, int> cropSize, Tuple<float, float> sideRatio,
            Tuple<float, float> areaRatio, Tuple<float, float> aspectRatio, string jitterType)
        {
            PairIntInt cropSizeSwig = new PairIntInt(cropSize.Item1, cropSize.Item2);
            PairFloatFloat sideRatioSwig = new PairFloatFloat(sideRatio.Item1, sideRatio.Item2);
            PairFloatFloat areaRatioSwig = new PairFloatFloat(areaRatio.Item1, areaRatio.Item2);
            PairFloatFloat aspectRatioSwig = new PairFloatFloat(aspectRatio.Item1, aspectRatio.Item2);
            return ReaderCrop(cropType, cropSizeSwig, sideRatioSwig, areaRatioSwig, aspectRatioSwig, jitterType);
        }

        public static CNTKDictionary ImageDeserializer(string fileName, string labelStreamName, uint numLabels, string imageStreamName, IList<CNTKDictionary> deserializers)
        {
            DictionaryVector deserializersSwig = Helper.AsDictionaryVector(deserializers);
            return ImageDeserializer(fileName, labelStreamName, numLabels, imageStreamName, deserializersSwig);
        }

        public static Function Convolution(Variable convolutionMap, Variable operand, NDShape strides, IEnumerable<bool> sharing, IEnumerable<bool> autoPadding)
        {
            BoolVector sharingVec = Helper.AsBoolVector(sharing);
            BoolVector autoPaddingVec = Helper.AsBoolVector(autoPadding);
            return CNTKLib.Convolution(convolutionMap, operand, strides, sharingVec, autoPaddingVec);
        }
    }
}
