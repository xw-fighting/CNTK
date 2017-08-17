using System.Collections.Generic;

namespace CNTK
{
    public partial class MinibatchSourceConfig
    {
        public MinibatchSourceConfig(IList<CNTKDictionary> deserializers) : this(Helper.AsDictionaryVector(deserializers))
        {
        }
    }
}
