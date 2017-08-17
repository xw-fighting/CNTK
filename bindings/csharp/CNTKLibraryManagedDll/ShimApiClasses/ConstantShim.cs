using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class Constant
    {
        public Constant(NDShape shape, float initValue, DeviceDescriptor device, string name = "") :
            this(shape, DataType.Float, initValue, device, name)
        {

        }
        /// <summary>
        /// Create a scalar constant. The specified value is cast to the specified DataType
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        static public Constant Scalar<T>(T value, DeviceDescriptor device)
        {
            if (device == null)
            {
                device = DeviceDescriptor.CPUDevice;
            }

            if (typeof(T).Equals(typeof(float)))
            {
                return _ScalarFloat((float)(object)value, device);
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                return _ScalarDouble((double)(object)value, device);
            }
            else
            {
                throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }
    }
}
