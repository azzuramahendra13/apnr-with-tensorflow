import openvino as ov

core = ov.Core()
devices = core.available_devices

for device in devices:
    device_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

classification_model_xml = "..\..\openvino_test\model\ir\my_model_mobilenet.xml"

model = core.read_model(model=classification_model_xml)
compiled_model = core.compile_model(model=model, device_name="CPU")

input_layer = model.input(0)
output_layer = model.outputs

print(input_layer)