from openvino.runtime import Core

core = Core()
for device in core.available_devices:
    name = core.get_property(device, "FULL_DEVICE_NAME")
    print(device, ":", name)