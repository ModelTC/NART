# An example of model conversion and inference using NART

## Step 1. Model conversion (PyTorch -> ONNX, PyTorch -> Caffe)

**To run the model conversion script of this example,
you need PyTorch LTS 1.8 and TorchVision.**

```sh
python model_conversion.py
```
or view the jupyter notebook `model_conversion.ipynb`.

The script `model_conversion` converts
the pretrained ResNet50 PyTorch model into either a Caffe model or an ONNX model.

You can edit the script to choose the destination model type.
Please check the script `model_conversion.py` or the jupyter notebook `model_conversion.ipynb` for details.

After run the python script or the jupyter notebook,
you can find the result model in `tmp` directory.

## Step 2. Convert the model to NART parade (ONNX -> NART, Caffe -> NART)

**This step requires installed NART python modules.**

To convert the ONNX model into NART parade that runs on CPU, run
```sh
python -m nart.switch -t default --onnx tmp/model.onnx --output tmp/engine.bin
```

NART switch can transform Caffe models and ONNX models into NART `parade`,
which contains weights and other essential information.

NART switch also takes information like the target NART case modules
that will run the `parade`.

If the NART case module `cuda` is built,
you can also convert the model into NART parade for GPU,
by replacing `-t default` to `-t cuda`.

If the result model in the previous step is a Caffe model,
replace `--onnx tmp/model.onnx`
to `--prototxt tmp/model.prototxt --model tmp/model.caffemodel`.

The result file `tmp/engine.bin` is the converted NART `parade` itself.
And the result file `tmp/engine.bin.json` is a configuration of workspaces to run the NART `parade`.

## Step 3. Inference

Now you can run the inference of ResNet50 only with the NART `parade` and the NART case runtime:
```sh
python run_inference.py
```
or view the jupyter notebook `run_inference.ipynb`.

`run_inference` is an example to run the `parade` with Python binding.
Please check the script `run_inference.py` or the jupyter notebook `run_inference.ipynb` for details.

### Benchmark

If NART tools are installed,
you can profile NART `parade` with `nart_promark`:
```sh
nart_promark -m tmp/engine.bin -c tmp/engine.bin.json
```

PS: if `nart_promark` cannot find `libart.so` or other shared library,
please try set the environment variable `LD_LIBRARY_PATH`.
