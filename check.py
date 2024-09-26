import pycuda.driver as cuda
import pycuda.autoinit

device = cuda.Device(0)

device_attributes = device.get_attributes()

if cuda.device_attribute.CONCURRENT_KERNELS in device_attributes:
    if device_attributes[cuda.device_attribute.CONCURRENT_KERNELS]:
        print("GPU supports concurrent streams.")
    else:
        print("GPU does not support concurrent streams.")
else:
    print("Could not determine concurrent streams support.")
