using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CNTKLibraryCSTrainingTest
{
#pragma warning disable CS0219 // Variable is assigned but its value is never used
    public class TransferLearning
    {
        static bool cifar = true;

        public static void TrainAndEvaluateTransferLearning(DeviceDescriptor device)
        {
            Function base_model0 = Function.Load("C:/LiqunWA/cntk/ForkSaved/ResNet_18.model", device);
            IList<Function> fN = base_model0.RootFunction.FindAllWithName("features");
            IList<Function> zN = base_model0.RootFunction.FindAllWithName("z.x");

            if (cifar)
            {

                string base_model_file = "C:/LiqunWA/cntk/ForkSaved/ResNet_18.model";
                string flowermapFile = "C:/LiqunWA/cntk/ForkSaved/train_map.txt";
                string feature_node_name = "features";
                string last_hidden_node_name = "z.x";
                string new_output_node_name = "prediction";
                int[] image_dims = new int[] { 224, 224, 3 };

                string flower_model_file = "FlowersTransferLearning.model";
                string flower_results_file = "FlowersPredictions.txt";
                int flower_num_classes = 102;

                string animal_model_file = "AnimalsTransferLearning.model";
                string animal_results_file = "AnimalsPredictions.txt";
                int animal_num_classes = 0; // ?

                bool freeze = true;

                var image_input = Variable.InputVariable(image_dims, DataType.Float, feature_node_name);
                var label_input = Variable.InputVariable(new int[] { flower_num_classes }, DataType.Float, "labels");

                Function base_model = Function.Load(base_model_file, device);

                var tl_model = create_model(base_model, feature_node_name, new_output_node_name, last_hidden_node_name,
                    flower_num_classes, image_input, device, freeze);

                TrainModelCifar(tl_model, image_input, label_input, image_dims, flower_num_classes, flowermapFile, device);
            }
            else
            {
                string base_model_file = "C:/LiqunWA/cntk/ForkSaved/CNTK_103D.model";
                string train_file = "C:/LiqunWA/cntk/ForkSaved/Train-28x28_cntk_text.txt";
                string test_file = "C:/LiqunWA/cntk/ForkSaved/Test-28x28_cntk_text.txt";
                train_file = test_file; // use small file to speed up dev
                string old_feature_node_name = "features";
                string new_feature_node_name = "new_features";
                string last_hidden_node_name = "second_max";
                string new_output_node_name = "prediction";
                int[] image_dims = new int[] { 28, 28, 1 };

                int new_model_num_classes = 10;

                bool freeze = true;

                var image_input = Variable.InputVariable(image_dims, DataType.Float, new_feature_node_name);
                var label_input = Variable.InputVariable(new int[] { new_model_num_classes }, DataType.Float, "outputs");

                Function base_model = Function.Load(base_model_file, device);

                PrintGraph(base_model.RootFunction, 0);
                var tl_model = create_model(base_model, old_feature_node_name, new_output_node_name, last_hidden_node_name,
                    new_model_num_classes, image_input, device, freeze);
                PrintGraph(tl_model.RootFunction, 0);

                TrainModel(tl_model, image_input, label_input, image_dims, new_model_num_classes, train_file, device);
            }
        }

        static MinibatchSource CreateCifarMinibatchSource(uint epochSize)
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
            config.maxSamples = epochSize;

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }

        public static Function create_model(Function base_model, string old_feature_node_name, string new_output_node_name,
            string last_hidden_node_name, int new_model_num_classes, Variable input_features, 
            DeviceDescriptor device, bool freeze = false)
        {
            Variable old_feature_node = base_model.Arguments.Single(a => a.Name == old_feature_node_name);
            Function last_node = base_model.FindByName(last_hidden_node_name);
            Variable new_feature_node = CNTKLib.PlaceholderVariable(input_features.Name);

            // Clone the desired layers with fixed weights
            // TODO: clone from last_node, not to combine and owner or at least hide it. 
            // Can the cloned model get its name? probably not because it is not different model.
            Function cloned_layers = Function.Combine(new List<Variable>() { ((Variable)last_node).Owner }).Clone(
                freeze ? ParameterCloningMethod.Freeze : ParameterCloningMethod.Clone,
                new Dictionary<Variable, Variable>() { { old_feature_node, new_feature_node } });

            Console.WriteLine("cloned_layers");
            PrintGraph(cloned_layers.RootFunction, 0);

            // Add new dense layer for class prediction
            // TODO: take a prefabricated feat_norm as input node. apply it when cloning to bypass placeholder/replace placeholder things
            Function feat_norm = CNTKLib.Minus(input_features, Constant.Scalar(DataType.Float, 114.0F));
            Function cloned_out = cloned_layers.ReplacePlaceholders(new Dictionary<Variable, Variable>() { { new_feature_node, feat_norm } });

            Console.WriteLine("cloned_out after ReplacePlaceholders applied to cloned_layers ");
            PrintGraph(cloned_out.RootFunction, 0);
            Function z = TestHelper.Dense(cloned_out, new_model_num_classes, device, TestHelper.Activation.None, new_output_node_name);

            Console.WriteLine("z");
            PrintGraph(z.RootFunction, 0);

            return z;
        }

        static void TrainModelCifar(Function tl_model, Variable image_input, Variable label_input,
            int[] image_dims, int num_classes, string train_map_file, DeviceDescriptor device)
        {

        }

        static void TrainModel(Function tl_model, Variable image_input, Variable label_input,
            int[] image_dims, int num_classes, string train_map_file,  DeviceDescriptor device)
        {

            // Create the minibatch source and input variables
            MinibatchSource minibatchSource = CreateMinbatchSource(train_map_file, image_dims, num_classes, "features", "labels");

            var featureStreamInfo = minibatchSource.StreamInfo("features");
            var labelStreamInfo = minibatchSource.StreamInfo("labels");

            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(tl_model, label_input);
            var predictionError = CNTKLib.ClassificationError(tl_model, label_input);

            // Instantiate the trainer object
            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.003125, TrainingParameterScheduleDouble.UnitType.Sample);

            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(tl_model.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(tl_model, trainingLoss, predictionError, parameterLearners);

            const uint minibatchSize = 64;
            const uint numSamplesPerSweep = 60000;
            const uint numSweepsToTrainWith = 2;
            const uint numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

            int outputFrequencyInMinibatches = 20;
            for (int i = 0; i < numMinibatchesToTrain; ++i)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                IList<MinibatchData> minibatchDatas = minibatchData.Values.ToList();


                Value featuresValue = TestHelper.MinibatchDataToValue(minibatchDatas[0], image_dims, (int)minibatchSize);
                Value labelsValue = TestHelper.MinibatchDataToValue(minibatchDatas[1], new int[]{ num_classes }, (int)minibatchSize);

                var arguments = new Dictionary<Variable, Value>
                {
                    { image_input, featuresValue },
                    { label_input, labelsValue }
                };

                trainer.TrainMinibatch(arguments, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            }
        }

#if false
        static MinibatchSource CreateMinbatchSource(string map_file, int[] image_dims, int num_classes, bool randomize = true)
        {
            DictionaryVector transforms = new DictionaryVector();
            transforms.Add(CNTKLib.ReaderScale(image_dims[2], image_dims[1], image_dims[0], "linear"));

            CNTKDictionary deserializer = CNTKLib.ImageDeserializer(map_file, "label", (uint)num_classes, "image", transforms);
            DictionaryVector deserializers = new DictionaryVector();
            deserializers.Add(deserializer);
            MinibatchSourceConfig minibatchSourceConfig = new MinibatchSourceConfig(deserializers, randomize);
            return CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }
#else
        static MinibatchSource CreateMinbatchSource(string map_file, int[] image_dims, int num_classes,
            string featureStreamName, string labelsStreamName, bool randomize = true)
        {
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[] 
            {
                new StreamConfiguration(featureStreamName, image_dims[0] * image_dims[1] * image_dims[2]),
                new StreamConfiguration(labelsStreamName, num_classes)
            };

            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(map_file, streamConfigurations);
            return minibatchSource;
        }
#endif

        static void PrintGraph(Function function, int spaces)
        {
            if (cifar)
                return;
            if (function.Inputs == null || function.Inputs.Count() == 0)
            {
                Console.WriteLine(new string('.', spaces) + "(" + function.Name + ")" + function.AsString());
                return;
            }
            
            foreach (var input in function.Inputs)
            {
                Console.WriteLine(new string('.', spaces) + "(" + function.Name + ")" + "->" + 
                    "(" + input.Name + ")" + input.AsString());
            }
            foreach (var input in function.Inputs)
            {
                if (input.Owner != null)
                {
                    Function f = input.Owner;
                    PrintGraph(f, spaces + 4);
                }
            }
        }
    }
}
