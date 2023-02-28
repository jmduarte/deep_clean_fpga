# Deep Clean FPGA Deployment

this is just a small repository containing files relevant for deploying deep clean on an alveo.

relevant files for simply running the model:
- `run_deep_clean.py`
- `deep_clean.xclbin` (located in same folder as above python file)

instructions:

set up vivado 2020.1 and install conda or mamba


check out code
```bash
git clone https://github.com/jmduarte/deep_clean_fpga
cd deep_clean_fpga
```

install python dependencies with `environment.yml`

```bash
conda env create -f environment.yml
conda activate deepclean
```


unzip model
```bash
unzip keras_deep_clean.zip
```

convert model
```bash
python simple_vivado_accel_deep_clean.py
```